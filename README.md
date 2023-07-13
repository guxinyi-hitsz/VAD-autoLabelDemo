# audio_detect
## Repository
https://github.com/Appen-China/audio_detect

## 依赖环境
基于speechbrain、malaya-speech、webrtcvad开源项目
```bash
pip install -r requirements.txt
```
## 参数说明

### crdnn-vad
详见crdnn\README.md

### webrtc-vad
详见webrtc\README.md

## Feature List
+ 20230224 语音端点检测的连续分段保存为Textgrid文件, 格式例如
```
File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0 
xmax = 67.2853125 
tiers? <exists> 
size = 1
item []: 
    item [1]:
        class = "IntervalTier" 
        name = "VoiceActivityDetect" 
        xmin = 0 
        xmax = 67.2853125 
        intervals: size = 15 
        intervals [1]:
            xmin = 0 
            xmax = 2.2689725526001703 
            text = "sil" 
        intervals [2]:
            xmin = 2.2689725526001703 
            xmax = 4.96040116800444 
            text = "active" 
        ...
        intervals [14]:
            xmin = 60.28722132596923 
            xmax = 65.77724409173217 
            text = "active" 
        intervals [15]:
            xmin = 65.77724409173217 
            xmax = 67.2853125 
            text = "sil"
```

在Praat上预览效果
![20230224161649](https://appen-pe.oss-cn-shanghai.aliyuncs.com/imgupload/20230224161649.png)

+ 20230301 audio_detect.webrtc.vad.py 的分段性能达到基准。
    - 高信噪比音频：
![20230301132656](https://appen-pe.oss-cn-shanghai.aliyuncs.com/imgupload/20230301132656.png)

    - 低信噪比音频：
![20230301132844](https://appen-pe.oss-cn-shanghai.aliyuncs.com/imgupload/20230301132844.png)

+ 20230310 audio_detect.crdnn.vad.py 集成了speechbrain开源的VAD深度学习预训练模型，相较于webrtcvad有更精准的端点和抗噪性能。
![20230313141005](https://appen-pe.oss-cn-shanghai.aliyuncs.com/imgupload/20230313141005.png)
![20230313142536](https://appen-pe.oss-cn-shanghai.aliyuncs.com/imgupload/20230313142536.png)

+ 20230321 audio_detect.webrtc.vad_parallel.py 开启多线程加速和无效分段合并，执行时间复杂度比crdnn.vad小，作为baseline方案。
![20230321135924](https://appen-pe.oss-cn-shanghai.aliyuncs.com/imgupload/20230321135924.png)