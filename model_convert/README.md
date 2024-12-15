# 模型转换

下载[checkpoint](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip)，解压到 `checkpoints_v2` 目录.

## 准备环境

```
conda create -n openvoice python=3.9
conda activate openvoice
pip install -r requirements.txt
```

## 导出 ONNX 模型及量化数据集

```
pip install git+https://github.com/myshell-ai/MeloTTS.git
python -m unidic download
python export_onnx.py
```

## 编译模型

```
pulsar2 build --input decoder.onnx --config config_decoder_u16.json --output_dir decoder --output_name decoder.axmodel --target_hardware AX650 --compiler.check 0
```
