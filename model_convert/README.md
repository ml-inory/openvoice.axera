# 模型转换

## 准备环境

```
conda create -n openvoice python=3.9
conda activate openvoice
pip install -r requirements.txt
```

## 导出 ONNX 模型及量化数据集

```
pip install git+https://github.com/myshell-ai/MeloTTS.git
# python -m unidic download
wget https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip
unzip checkpoints_v2_0417.zip
python export_onnx.py
```

export_onnx.py 运行参数说明:  
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| --encoder | 输出encoder路径 | ./encoder.onnx |
| --decoder | 输出decoder路径 | ./decoder.onnx |
| --enc_len | encoder输入长度 | 1024 |
| --dec_len | decoder输入长度 | 128 |

## 编译模型

Encoder:  
```
pulsar2 build --input encoder.onnx --config config_encoder_u16.json --output_dir encoder --output_name encoder.axmodel --target_hardware AX650 --compiler.check 0
```

Decoder:  
```
pulsar2 build --input decoder.onnx --config config_decoder_u16.json --output_dir decoder --output_name decoder.axmodel --target_hardware AX650 --compiler.check 0
```

转换完成后复制axmodel到models目录下。
