# sp_stats 基于时域的音频统计量


| 字段名         | 含义                                        | 计算逻辑                             |
| ---------------- | --------------------------------------------- | -------------------------------------- |
| fName          | 文件名                                      |                                      |
| sampRate       | 音频采样率                                  | 文件头                               |
| sampBits       | 采样位宽，表示一个采样点的存储长度          | 文件头                               |
| numChan        | 声道数                                      | 文件头                               |
| chanNum        | 单声道选择，默认0                           | 执行参数                             |
| numSamp        | 采样点数                                    | Number of samples                    |
| numSec         | 音频总时长                                  | numSec = numSamp / sampRate          |
| blockSizeMs    | 音频分帧长度，默认10ms                      | 执行参数                             |
| smoothLen      | 默认10                                      | 执行参数                             |
| sigSec         | 有效语音时长，各个语音段时长求和            |                                      |
| bkgSec         | 背景音时长，各个静音段时长求和              |                                      |
| sigRatio       | 有效语音时长占比                            | sigRatio = sigSec / numSec           |
| startSilSec    | 起始静音段时长                              |                                      |
| endSilSec      | 结尾静音段时长                              |                                      |
| endSpSec       | 语音段结尾时刻（秒）                        | endSpSec = numSec - endSilSec        |
| dcOffsetdB     | 直流分量（分贝）                            | np.mean(inWavVec, axis=0)            |
| meanPowdB      | 平均能量（分贝）                            | np.mean(np.square(inWavVec), axis=0) |
| lufsdBK        | loudness units relative to full scale，响度 |                                      |
| maxBlockPowdB  |                                             |                                      |
| blockRangedB   |                                             |                                      |
| medBlockPowdB  |                                             |                                      |
| stdBlockPowdB  |                                             |                                      |
| maxConvPowdB   |                                             |                                      |
| minConvPowdB   |                                             |                                      |
| convRangedB    |                                             |                                      |
| sigRmsdB       | 语音RMS                                     |                                      |
| sigRmsdBA      | A-weight语音RMS                             |                                      |
| bkgRmsdB       | 背景音RMS                                   |                                      |
| bkgRmsdBA      | A-weight背景音RMS                           |                                      |
| sigThreshdB    |                                             |                                      |
| SNRdB          | 信噪比分贝                                  | SNRdB=sigRmsdB-bkgRmsdB              |
| SNRdBA         | A-weight信噪比分贝                          | SNRdBA=sigRmsdBA-bkgRmsdBA           |
| startPowdB     |                                             |                                      |
| endPowdB       |                                             |                                      |
| maxStSilBlkdB  |                                             |                                      |
| maxEdSilBlkdB  |                                             |                                      |
| posPeakInt     | 采样振幅峰值（+）                           |                                      |
| negPeakInt     | 采样振幅峰值（-）                           |                                      |
| absPeakInt     | 采样振幅峰值（绝对值）                      |                                      |
| absPeakPerc    |                                             |                                      |
| absPeakdB      |                                             |                                      |
| altPeakdB      |                                             |                                      |
| avPeakdB       |                                             |                                      |
| numPeaks       |                                             |                                      |
| numPkRuns      |                                             |                                      |
| pkRunAvLen     |                                             |                                      |
| numZeros       |                                             |                                      |
| zeroRunLenMax  |                                             |                                      |
| zeroRunLenMean |                                             |                                      |
| satKur         |                                             |                                      |
| satCog         |                                             |                                      |
| satR1          |                                             |                                      |
| satR2          |                                             |                                      |
|                |                                             |                                      |
