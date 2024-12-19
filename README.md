# OpenVoice.Axera

人声克隆，官方repo: https://github.com/myshell-ai/OpenVoice.git

## 模型转换

参考[模型转换文档](model_convert/README.md)

## 安装依赖
```
pip3 install -r requirements.txt
```
由于默认的安装路径可能空间不足，此时可以使用--prefix参数将依赖安装在其它路径，如：  

```
pip3 install -r requirements.txt --prefix=/root/site-packages
```

其中prefix可以替换为其他路径，但需要确保其在PYTHONPATH以及PATH环境变量中。
环境变量配置示例如下：
```
vim /root/.bashrc  

在最后添加
export PYTHONPATH=$PYTHONPATH:/opt/site-packages/local/lib/python3.10/dist-packages:/root/site-packages/local/lib/python3.10/dist-packages  
export PATH=$PATH:/opt/site-packages/local/bin:/root/site-packages/local/bin

保存退出编辑后
source /root/.bashrc
```


## 运行
```
python3 main.py -i 输入音频 -o 输出音频(默认为output.wav)
```
所有运行参数：  
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| -i | 输入音频，wav格式 | 无 |
| -o | 输出音频，wav格式 | output.wav |
| -e/--encoder | encoder模型路径 | ../models/encoder.axmodel |
| -d/--decoder | decoder模型路径 | ../models/decoder.axmodel |
| --g_src | 源人声特征值，bin格式 | ../models/g_src.bin |
| --g_dst | 目标人声特征值，bin格式 | ../models/g_dst.bin |
| --enc_len | encoder输入长度 | 1024 |
| --dec_len | decoder输入长度 | 128 |