import numpy as np
import argparse
from swig_decoders import map_batch,ctc_beam_search_decoder_batch,TrieVector, PathTrie
import multiprocessing
import torchaudio.compliance.kaldi as kaldi
import torch, torchaudio
import yaml
import os
from axengine import InferenceSession


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, required=True, help="Input audio file")
    parser.add_argument("--online", required=True, action="store_true")

    parser.add_argument("--config", type=str, default="config.yaml", help="yaml file in checkpoint path")
    parser.add_argument("--encoder_online", type=str, default="axmodel/encoder_online.axmodel")
    parser.add_argument("--encoder_offline", type=str, default="axmodel/encoder_offline.axmodel")
    parser.add_argument("--decoder", type=str, default="axmodel/decoder.axmodel")
    parser.add_argument("--offline_seq_len", type=int, default=1024)
    parser.add_argument("--online_seq_len", type=int, default=67)
    parser.add_argument("--decoder_len", type=int, default=32)
    parser.add_argument('--mode',
                        choices=[
                            'ctc_greedy_search', 'ctc_prefix_beam_search', 'attention_rescoring'],
                        default='ctc_prefix_beam_search',
                        help='decoding mode')
    return parser.parse_args()


def compute_feats(audio_file: str, sr=16000) -> torch.Tensor:
    waveform, sample_rate = torchaudio.load(audio_file, normalize=False)
    waveform = waveform.to(torch.float)
    if sample_rate != sr:
        waveform = torchaudio.transforms.Resample(
            orig_freq=sample_rate, new_freq=sr)(waveform)
    # NOTE (MengqingCao): complex dtype not supported in torch_npu.abs() now,
    # thus, delay placing data on NPU after the calculation of fbank.
    # revert me after complex dtype is supported.
    feats = kaldi.fbank(waveform,
                        num_mel_bins=80,
                        frame_length=25,
                        frame_shift=10,
                        energy_floor=0.0,
                        sample_frequency=sr)
    feats = feats.unsqueeze(0)
    return feats.numpy()


def ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, vocabulary, mode='ctc_prefix_beam_search'):
    beam_size = beam_log_probs.shape[-1]
    batch_size = beam_log_probs.shape[0]
    num_processes = min(multiprocessing.cpu_count(), batch_size)
    hyps = []
    score_hyps = []
    
    if mode == 'ctc_greedy_search':
        if beam_size != 1:
            log_probs_idx = beam_log_probs_idx[:, :, 0]
        batch_sents = []
        for idx, seq in enumerate(log_probs_idx):
            batch_sents.append(seq[0:encoder_out_lens[idx]].tolist())
        hyps = map_batch(batch_sents, vocabulary, num_processes,
                         True, 0)
    elif mode in ('ctc_prefix_beam_search', "attention_rescoring"):
        batch_log_probs_seq_list = beam_log_probs.tolist()
        batch_log_probs_idx_list = beam_log_probs_idx.tolist()
        batch_len_list = encoder_out_lens.tolist()
        batch_log_probs_seq = []
        batch_log_probs_ids = []
        batch_start = []  # only effective in streaming deployment
        batch_root = TrieVector()
        root_dict = {}
        for i in range(len(batch_len_list)):
            num_sent = batch_len_list[i]
            batch_log_probs_seq.append(
                batch_log_probs_seq_list[i][0:num_sent])
            batch_log_probs_ids.append(
                batch_log_probs_idx_list[i][0:num_sent])
            root_dict[i] = PathTrie()
            batch_root.append(root_dict[i])
            batch_start.append(True)
        score_hyps = ctc_beam_search_decoder_batch(batch_log_probs_seq,
                                                   batch_log_probs_ids,
                                                   batch_root,
                                                   batch_start,
                                                   beam_size,
                                                   num_processes,
                                                   0, -2, 0.99999)
        if mode == 'ctc_prefix_beam_search': 
            for cand_hyps in score_hyps:
                hyps.append(cand_hyps[0][1])
            hyps = map_batch(hyps, vocabulary, num_processes, False, 0)
    return hyps, score_hyps


def pad_array_along_axis(array, pad_width, axis, mode='constant', **kwargs):
    if array.shape[axis] < pad_width:
        # 创建全零填充宽度元组列表
        full_pad_width = [(0, 0)] * array.ndim
        
        # 更新指定轴上的填充宽度
        full_pad_width[axis] = (0, pad_width - array.shape[axis])

        # 执行填充操作
        return np.pad(array, pad_width=full_pad_width, mode=mode, **kwargs)
    else:
        return array


def run_ort():
    args = get_args()
    print(f"online: {args.online}")
    print(f"mode: {args.mode}")
    print(f"calib_data_path: {args.calib_data_path}")
    calib_data_path = args.calib_data_path
    if calib_data_path is not None:
        os.makedirs(calib_data_path, exist_ok=True)

    with open(args.config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)

    feats = compute_feats(args.input)
    # Load dict
    vocabulary = []
    char_dict = {}
    with open("vocab.txt", 'r') as fin:
        for line in fin:
            arr = line.strip().split()
            assert len(arr) == 2
            char_dict[int(arr[1])] = arr[0]
            vocabulary.append(arr[0])
    eos = sos = len(char_dict) - 1

    decoder = None
    if args.mode == 'attention_rescoring':
        decoder = InferenceSession.load_from_model(args.decoder)

    if not args.online:
        seq_len = args.offline_seq_len
        encoder = InferenceSession.load_from_model(args.encoder_offline)
        feats = feats[:, :seq_len, :]
        speech_lengths = np.array([feats.shape[1]], dtype=np.int32)
        if feats.shape[1] < seq_len:
            feats = pad_array_along_axis(feats, pad_width=seq_len, axis=1)

        outputs = encoder.run(input_feed={"speech": feats, "speech_lengths": speech_lengths})
        encoder_out, encoder_out_lens, ctc_log_probs, beam_log_probs, beam_log_probs_idx = \
            outputs["encoder_out"], outputs["encoder_out_lens"], outputs["ctc_log_probs"], outputs["beam_log_probs"], outputs["beam_log_probs_idx"]

        results, _ = ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, vocabulary, args.mode)
        if len(results):
            result = "".join(results)
    else:
        seq_len = args.online_seq_len
        encoder = InferenceSession.load_from_model(args.encoder_online)
        batch_size = 1
        subsampling = 4
        context = 7
        decoding_chunk_size = 16
        num_decoding_left_chunks = 5

        stride = subsampling * decoding_chunk_size
        decoding_window = (decoding_chunk_size - 1) * subsampling + context        
        required_cache_size = decoding_chunk_size * num_decoding_left_chunks

        output_size = configs["encoder_conf"]["output_size"]
        num_layers = configs["encoder_conf"]["num_blocks"]
        cnn_module_kernel = configs["encoder_conf"].get("cnn_module_kernel", 1) - 1
        head = configs["encoder_conf"]["attention_heads"]
        d_k = configs["encoder_conf"]["output_size"] // head
                    
        att_cache = np.zeros((batch_size, num_layers, head, required_cache_size, d_k * 2), dtype=np.float32)
        cnn_cache = np.zeros((batch_size, num_layers, output_size, cnn_module_kernel), dtype=np.float32)
        cache_mask = np.zeros((batch_size, 1, required_cache_size), dtype=np.float32)
        offset = np.zeros((batch_size, 1), dtype=np.int32)

        encoder_out = []
        beam_log_probs = []
        beam_log_probs_idx = []

        num_frames = feats.shape[1]
        result = ""
        for cur in range(0, num_frames - context + 1, stride):
            end = min(cur + decoding_window, num_frames)
            chunk_xs = feats[:, cur:end, :]
            if chunk_xs.shape[1] < decoding_window:
                chunk_xs = pad_array_along_axis(chunk_xs, pad_width=decoding_window, axis=1)
                chunk_xs = chunk_xs.astype(np.float32)
            chunk_lens = np.full(batch_size, fill_value=chunk_xs.shape[1], dtype=np.int32)

            encoder_input = {"chunk_lens": chunk_lens, "att_cache": att_cache, "cnn_cache": cnn_cache, 
                            "chunk_xs": chunk_xs, "cache_mask": cache_mask, "offset": offset}
            
            outputs = encoder.run(input_feed=encoder_input)
            chunk_log_probs, chunk_log_probs_idx, chunk_out, chunk_out_lens, offset, att_cache, cnn_cache, cache_mask = \
                outputs["chunk_log_probs"], outputs["chunk_log_probs_idx"], outputs["chunk_out"], \
                outputs["chunk_out_lens"], outputs["offset"], outputs["att_cache"], \
                outputs["cnn_cache"], outputs["cache_mask"]
            
            chunk_log_probs_idx = chunk_log_probs_idx.astype(np.int32)
            chunk_out_lens = chunk_out_lens.astype(np.int32)

            encoder_out.append(chunk_out)
            beam_log_probs.append(chunk_log_probs)
            beam_log_probs_idx.append(chunk_log_probs_idx)
            
            # ctc decode
            chunk_hyps, _ = ctc_decoding(chunk_log_probs, chunk_log_probs_idx, chunk_out_lens, vocabulary, args.mode)
            # print(chunk_hyps)
            if len(chunk_hyps):
                result = "".join(chunk_hyps)

        encoder_out = np.concatenate(encoder_out, axis=1)
        encoder_out_lens = np.full(batch_size, fill_value=encoder_out.shape[1], dtype=np.int32)
        beam_log_probs = np.concatenate(beam_log_probs, axis=1)
        beam_log_probs_idx = np.concatenate(beam_log_probs_idx, axis=1)

    if args.mode == 'attention_rescoring':
        IGNORE_ID = -1
        beam_size = beam_log_probs.shape[-1]
        batch_size = beam_log_probs.shape[0]
        num_processes = min(multiprocessing.cpu_count(), batch_size)
        hyps, score_hyps = ctc_decoding(beam_log_probs, beam_log_probs_idx, encoder_out_lens, vocabulary, args.mode)
        ctc_score, all_hyps = [], []
        max_len = 0
        for hyps in score_hyps:
            cur_len = len(hyps)
            if len(hyps) < beam_size:
                hyps += (beam_size - cur_len) * [(-float("INF"), (0,))]
            cur_ctc_score = []
            for hyp in hyps:
                cur_ctc_score.append(hyp[0])
                all_hyps.append(list(hyp[1]))
                if len(hyp[1]) > max_len:
                    max_len = len(hyp[1])
            ctc_score.append(cur_ctc_score)
            ctc_score = np.array(ctc_score, dtype=np.float32)
        max_len = args.decoder_len - 2
        hyps_pad_sos_eos = np.ones(
            (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
        r_hyps_pad_sos_eos = np.ones(
            (batch_size, beam_size, max_len + 2), dtype=np.int64) * IGNORE_ID
        hyps_lens_sos = np.ones((batch_size, beam_size), dtype=np.int32)
        k = 0
        for i in range(batch_size):
            for j in range(beam_size):
                cand = all_hyps[k]
                l = len(cand) + 2
                hyps_pad_sos_eos[i][j][0:l] = [sos] + cand + [eos]
                r_hyps_pad_sos_eos[i][j][0:l] = [sos] + cand[::-1] + [eos]
                hyps_lens_sos[i][j] = len(cand) + 1
                k += 1

        if args.decoder_len > encoder_out.shape[1]:
            encoder_out = np.pad(encoder_out, [(0, 0),(0, args.decoder_len - encoder_out.shape[1]), (0, 0)], mode='constant', constant_values=0)
        elif args.decoder_len < encoder_out.shape[1]:
            encoder_out = encoder_out[:, :args.decoder_len, :]

        hyps_pad_sos_eos = hyps_pad_sos_eos.astype(np.int32)
        r_hyps_pad_sos_eos = r_hyps_pad_sos_eos.astype(np.int32)

        encoder_out_lens = np.full(batch_size, fill_value=args.decoder_len, dtype=np.int32)
        decoder_input = {
            "encoder_out": encoder_out,
            "encoder_out_lens": encoder_out_lens,
            "hyps_pad_sos_eos": hyps_pad_sos_eos,
            "hyps_lens_sos": hyps_lens_sos,
            "r_hyps_pad_sos_eos": r_hyps_pad_sos_eos,
            "ctc_score": ctc_score
        }

        best_index = decoder.run(input_feed=decoder_input)["best_index"]
        best_index = best_index.astype(np.int32)

        best_sents = []
        k = 0
        for idx in best_index:
            cur_best_sent = all_hyps[k: k + beam_size][idx]
            best_sents.append(cur_best_sent)
            k += beam_size
        hyps = map_batch(best_sents, vocabulary, num_processes)
        result = "".join(hyps)

    print(f"ASR Result: {result}")

if __name__ == "__main__":
    run_ort()