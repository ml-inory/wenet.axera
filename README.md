# wenet.axera
WeNet on Axera

转换好的模型请从[这里](https://github.com/ml-inory/wenet.axera/releases/tag/v1.0)下载  

## 运行ONNX
```
python run_ort.py -i demo.wav --config config.yaml --mode ctc_greedy_search/ctc_prefix_beam_search/attention_rescoring
```

## 运行axmodel
```
python3 run_ax.py -i demo.wav --config config.yaml --mode ctc_greedy_search/ctc_prefix_beam_search/attention_rescoring
```

---
如要自行导出，请参考以下步骤:  
## 下载预训练权重
从[这里](https://github.com/wenet-e2e/wenet/blob/main/docs/pretrained_models.md)选择一个Checkpoint model下载  

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
