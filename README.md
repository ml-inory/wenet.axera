# wenet.axera
WeNet on Axera

## ONNX导出
```
python export_onnx.py --config 20210601_u2++_conformer_exp_aishell/train.yaml --checkpoint 20210601_u2++_conformer_exp_aishell/final.pt
```

pulsar2 build --input onnx_model/encoder_offline.onnx --config config_encoder.json --output_dir axmodel/encoder --output_name encoder_offline.axmodel --target_hardware AX650