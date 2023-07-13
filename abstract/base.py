# -*- coding: UTF-8 -*- 

"""
Author  : Xinyi Gu
Date    : 2023/3/17 16:29
Change History: 
    @change
"""
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
import torchaudio
import textgrid
import uuid
import json
import os
import requests
import logging

logging.basicConfig()
logging.getLogger().setLevel(logging.ERROR)


__all__ = ['VADBase']


class VADBase(ABC):

    def __init__(self,
                 save_dir='./vad_results',
                 audio_dir='./vad_cache',
                 clean_history=False,
                 save_json=False,
                 save_continuous=False
                 ):
        self.option = {
            "save_dir": save_dir,
            "audio_dir": audio_dir,
            "clean_history": clean_history,
            "save_format": '.json' if save_json else '.TextGrid',
            "is_continuous": save_continuous
        }

        """containing the start second (or sample) of speech segments
            in even positions and their corresponding end in odd positions
            (e.g, [1.0, 1.5, 5.0, 6.0] means that we have two speech segment;
             one from 1.0 to 1.5 seconds and another from 5.0 to 6.0 seconds).
        """
        self.speech_boundaries = []
        self.silence_boundaries = []

        # clean history files if need
        if self.option["clean_history"]:
            if os.path.exists(self.option["audio_dir"]):
                shutil.rmtree(self.option["audio_dir"])
            if os.path.exists(self.option["save_dir"]):
                shutil.rmtree(self.option["save_dir"])
            # crdnn存sym-link的目录
            if os.path.exists('./pretrained_model_checkpoints'):
                shutil.rmtree('./pretrained_model_checkpoints')

    @abstractmethod
    def __call__(self,
                 wav_path,
                 use_channel=0,
                 large_chunk_sec=5,
                 small_chunk_sec=0.2,
                 min_adjacent_sec=0.25,
                 min_boundary_sec=0.25):
        pass

    def fetch_audio(self, url, dest):
        response = requests.get(url)
        assert response.status_code == 200, f"下载失败: {url}"

        os.makedirs(Path(dest).parent, exist_ok=True)
        with open(dest, 'wb') as fw:
            fw.write(requests.get(url).content)
        return dest

    def prepare_audio(self, wav_path, use_channel=0):
        """override
        """
        # Fetch audio file from web if not local
        if wav_path.startswith("http:") or wav_path.startswith("https:"):
            audio_file = self.fetch_audio(url=wav_path, dest=self.option["audio_dir"] + '/' + Path(wav_path).name)
        else:
            assert os.path.exists(wav_path) and Path(wav_path).suffix in ['.mp3', '.wav'], f"音频路径错误：{wav_path}"
            audio_file = Path(wav_path).absolute()

        # Select single channel data
        waveform, sr = torchaudio.load(audio_file)
        if waveform.ndim == 2:
            num_channels = waveform.shape[0]
            if num_channels > 1:
                waveform = waveform[use_channel].squeeze(0)
            else:
                waveform = waveform.squeeze(0)
        wav_vector = waveform.numpy()
        return wav_vector, sr

    def save_boundaries(self, filename):
        os.makedirs(self.option["save_dir"], exist_ok=True)
        save_path = self.option["save_dir"] + "/" + filename
        if self.option["save_format"] == '.json':
            self.save_boundaries_to_json(save_path)
        elif self.option["save_format"] == '.TextGrid':
            self.save_boundaries_to_textgrid(save_path)

    def sort_boundaries(self):
        boundaries = []
        for i in range(0, len(self.silence_boundaries), 2):
            boundaries.append([self.silence_boundaries[i:i + 2], False])
        for j in range(0, len(self.speech_boundaries), 2):
            boundaries.append([self.speech_boundaries[j:j + 2], True])
        # 按时间戳的起点升序
        s_boundaries = sorted(boundaries, key=lambda x: x[0][0], reverse=False)
        return s_boundaries

    def save_boundaries_to_textgrid(self, save_path):
        tg = textgrid.TextGrid()
        vad_tier = textgrid.IntervalTier(name='语音端点检测层')

        BOOLDICT = {
            True: 'active',
            False: 'sil'
        }

        for segment, segmentType in self.sort_boundaries():
            vad_tier.add(minTime=segment[0], maxTime=segment[1], mark=BOOLDICT[segmentType])
        tg.append(vad_tier)
        tg.write(save_path)
        print(f"＜（＾－＾）＞语音端点检测已生成。 {save_path}")

    def save_boundaries_to_json(self, save_path):
        """
            导出自定义模板的json格式
            { "segments": [
                {
                "_id": gen_uuid(),
                "start":round(start,2),
                "end":round(start+duration,2)
                },
                {...}
                ]
            }
        """
        segments_data = []
        for segment, segmentType in self.sort_boundaries():
            # 独立分段（仅speech）或连续分段（speech和silence）
            if not self.option["is_continuous"] and not segmentType:
                continue
            segment_dict = {
                "_id": self.gen_uuid(),
                "start": segment[0],
                "end": segment[1]
            }
            segments_data.append(segment_dict)
        res = {"segments": segments_data}
        with open(save_path, mode='w', encoding='utf-8') as fw:
            json.dump(res, fw, indent=4, ensure_ascii=False)
        print(res)

    @staticmethod
    def gen_uuid():
        uid = str(uuid.uuid4())
        suid = ''.join(uid.split('-'))
        return suid
