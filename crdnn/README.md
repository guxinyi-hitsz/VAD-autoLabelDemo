## 1. 安装依赖
```bash
pip install speechbrain
```

## Highlight
### 2. OSError('symbolic link privilege not held')
源码对yaml文件创建了软链接，需要用管理员权限运行。

### 3. Praat软件预览切分效果
如果原始音频音量很低，导入文件后，点击Info菜单查看声压峰值例如：
Total energy: 0.00220098726 Pascal² sec
这个声压较低，点击 Modify -> Scale Intensity 菜单输入新的平均音量强度，默认值是70dB SPL，可以使新的均方根振幅比
假设的听觉阈值0.00002 Pa高出70 dB。
如果声压低于-1Pa或高于+1Pa,可以正常被听见，此时再提高音量，可能会产生失真，新的平均音量强度应设置为60dB或更低。

## 命令行参数
```bash
# 导出为TextGrid
python vad.py -i https://projecteng.oss-cn-shanghai.aliyuncs.com/xgu_test/audio-detect-test/3659/1_channel_20211023-DE-de-DE0010-T10-S01.wav -c 0 -l 5 -s 0.5 -m 0.25 -b 0.25 --clean
```

```bash
# 导出为json, 语音端点为秒级，2位小数
python vad.py -i https://projecteng.oss-cn-shanghai.aliyuncs.com/xgu_test/audio-detect-test/3659/1_channel_20211023-DE-de-DE0010-T10-S01.wav -c 0 -l 5 -s 0.5 -m 0.25 -b 0.25 --clean --jsonOut --continuous
```
+ -i 音频的本地路径或URL
+ -c 选择的声道（用于多声道文件，默认为0）
+ -l 音频块读取长度（秒）
+ -s 音频块推理长度（秒）>=0.1，读取长度应为推理长度的整数倍，
+ -m 临近语音段的合并间隔（秒），当2个语音段间隔的静音段长度小于该阈值时，静音段转为语音段合并
+ -b 输出分段的最小长度（秒），当分段的长度小于该阈值时，合并
+ --clean 运行前删除pretraind_model_checkpoints目录下的历史文件
+ --jsonOut 保存为json格式，默认为TextGrid格式
+ --continuous 保存为json格式时，选择以连续分段或独立分段（只保留人声区间）

## 参数Tricks

+ -s至少为0.1秒，否则会报错
+ -s越小，分段越多，短音频设置小一些，长音频可设置大一些
+ -l设为-s的整数倍，

