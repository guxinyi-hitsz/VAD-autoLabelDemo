## 命令行参数
```bash
# 导出为TextGrid
python vad_parallel.py -i C:\Users\xgu\Documents\AudioDev\testData\longaudio\app_4107_6213_phnd_deb-1.wav -l 5 -s 0.02 -m 1.0 -b 0.5 --loop 15
```

```bash
# 导出为json
python vad.py -i https://projecteng.oss-cn-shanghai.aliyuncs.com/xgu_test/audio-detect-test/3659/1_channel_20211023-DE-de-DE0010-T10-S01.wav -l 5 -s 0.02 -m 1.0 -b 0.5 --loop 5 --jsonOut --continuous
```
+ -i 音频的本地路径或URL
+ -l 音频批处理长度（秒）
+ -s 音频帧长度（秒），范围在[0.01,0.03]之间，也就是10ms~30ms
+ -m 临近语音段的合并间隔（秒），当2个语音段间隔的静音段长度小于该阈值时，静音段转为语音段合并
+ -b 输出语音段的最小长度（秒），当语音段的长度小于该阈值时，语音段转为静音段合并
+ --loop  VAD处理无效分段的执行轮次，根据实际的数据调节，数字越大效果越好，默认15
+ --jsonOut （选填）保存为json格式，默认为TextGrid格式
+ --continuous （选填）保存为json格式时，选择以连续分段或独立分段（只保留人声区间）
+ -o （选填）输出目录，默认'./vad_results'
+ -c （选填）选择的声道（用于多声道文件，默认为0）
+ --clean （选填）运行前删除'./vad_cache'、输出目录下的历史文件
+ --mode （选填）VAD检测模式, 从[0,1,2,3]中选择, 数字越大, 检测为非人声的概率越大
