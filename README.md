# OpenVoice.Axera

人声克隆，官方repo: https://github.com/myshell-ai/OpenVoice.git

## 模型转换

参考[模型转换文档](model_convert/README.md)

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