# wenet.axera
WeNet on Axera

## ONNX导出
```
python export_onnx.py --config 20210601_u2++_conformer_exp_aishell/train.yaml --checkpoint 20210601_u2++_conformer_exp_aishell/final.pt
```
生成onnx_model目录

## axmodel导出
encoder_offline  
```
pulsar2 build --input onnx_model/encoder_offline.onnx --config config_encoder_offline.json --output_dir axmodel/encoder_offline --output_name encoder_offline.axmodel --target_hardware AX650
```

encoder_online 
```
pulsar2 build --input onnx_model/encoder_online.onnx --config config_encoder_online.json --output_dir axmodel/encoder_online --output_name encoder_online.axmodel --target_hardware AX650
```

decoder
```
pulsar2 build --input onnx_model/decoder.onnx --config config_decoder.json --output_dir axmodel/decoder --output_name decoder.axmodel --target_hardware AX650
```

## 运行ONNX
```
python run_ort.py -i demo.wav --config config.yaml --mode ctc_greedy_search/ctc_prefix_beam_search/attention_rescoring
```

## 运行axmodel
```
python3 run_ax.py -i demo.wav --config config.yaml --mode ctc_greedy_search/ctc_prefix_beam_search/attention_rescoring
```