# 模型转换

下载[checkpoint](https://myshell-public-repo-host.s3.amazonaws.com/openvoice/checkpoints_v2_0417.zip)，解压到`checkpoints_v2`目录.

```
conda create -n openvoice python=3.9
conda activate openvoice
pip install -r requirements.txt

python export_onnx.py
```