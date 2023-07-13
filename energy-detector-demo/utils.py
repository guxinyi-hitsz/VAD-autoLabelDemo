# -*- coding: UTF-8 -*- 
from pydub import AudioSegment
import array
import numpy as np


def to_numpy(arr: array.array):
    TRANS_ARRAY_TYPES = {
        'b': np.int8,
        'h': np.int16,
        'i': np.int32,
    }
    return np.array(arr.tolist(), dtype=TRANS_ARRAY_TYPES[arr.typecode])


def read_wav(file, chan=0):
    """读取音频数据，提取单声道归一化浮点向量
    - sound 原始音频对象，包含音频文件信息
    - ys    单声道numpy数组，按音频采样位宽转换为浮点值
    """
    sound = AudioSegment.from_wav(file)

    n = sound.channels
    chan = min(chan, n - 1)

    ys = to_numpy(arr=sound.get_array_of_samples())
    ys = ys[chan::n]
    
    intToFloatScalingFactor = 2**(sound.sample_width * 8-1)
    ys = ys.astype(float) / intToFloatScalingFactor
    
    return sound, ys
