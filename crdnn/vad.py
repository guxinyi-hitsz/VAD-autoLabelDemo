# -*- coding: UTF-8 -*- 

"""
Author  : Xinyi Gu
Date    : 2023/3/27 11:18 
Project :
Description:

https://huggingface.co/speechbrain/vad-crdnn-libriparty

Change History: 
    @change
"""
import sys
import os
from argparse import ArgumentParser

from pathlib import Path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from abstract import VADBase
from speechbrain.pretrained import VAD
import torchaudio
from torchaudio.functional import resample
from torch import from_numpy
import profile

class CRDnnVAD(VADBase):

    def __init__(self,
                 save_dir='./vad_results',
                 audio_dir='./vad_cache',
                 clean_history=False,
                 save_json=False,
                 save_continuous=False
                 ):
        super().__init__(save_dir, audio_dir, clean_history, save_json, save_continuous)

        # 从远程仓库拉取预训练模型文件，保存到本地
        self.model = VAD.from_hparams(run_opts={"device": "cpu"}, source="speechbrain/vad-crdnn-libriparty",
                                      savedir="pretrained_models/vad-crdnn-libriparty")

    def __call__(self,
                 wav_path,
                 use_channel=0,
                 large_chunk_sec=5,
                 small_chunk_sec=0.2,
                 min_adjacent_sec=0.25,
                 min_boundary_sec=0.25):
        local_audio, duration_sec = self.prepare_audio(audio_path=wav_path, use_channel=use_channel)

        boundaries_torch = self.model.get_speech_segments(audio_file=local_audio,
                                                          large_chunk_size=large_chunk_sec,
                                                          small_chunk_size=small_chunk_sec,
                                                          close_th=min_adjacent_sec,
                                                          len_th=min_boundary_sec,
                                                          overlap_small_chunk=True,
                                                          apply_energy_VAD=False)

        # boundaries_torch = self.model.get_speech_segments(audio_file=local_audio,
        #                                                   large_chunk_size=large_chunk_sec,
        #                                                   small_chunk_size=small_chunk_sec,
        #                                                   close_th=min_adjacent_sec,
        #                                                   len_th=min_boundary_sec,
        #                                                   overlap_small_chunk=True,
        #                                                   apply_energy_VAD=True,
        #                                                   en_activation_th=0.5,
        #                                                   en_deactivation_th=0.05)

        self.get_boundaries(boundaries_torch, duration_sec)

        self.save_boundaries(filename=str(Path(wav_path).stem) + self.option["save_format"])

    def prepare_audio(self, audio_path, use_channel=0):
        """override
        The speechbrain expects input recordings sampled at 16kHz (single channel)"""
        wav_vector, sr = super().prepare_audio(audio_path, use_channel)

        # vad模型的hparams定义了输入的采样率
        new_sr = self.model.sample_rate
        if sr != new_sr:
            print(
                f"Resampling audio from {sr} Herz to {new_sr} Hz"
            )
            waveform = resample(
                from_numpy(wav_vector).unsqueeze(0),
                sr,
                new_sr,
                lowpass_filter_width=64,
                rolloff=0.95,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            waveform = from_numpy(wav_vector).unsqueeze(0)

        duration_sec = waveform.shape[1] / new_sr

        os.makedirs(self.option["audio_dir"], exist_ok=True)
        local_audio_path = self.option["audio_dir"] + '/' + Path(audio_path).name
        audio_format = Path(audio_path).suffix.strip('.')
        torchaudio.save(local_audio_path, waveform, new_sr, format=audio_format)

        print(f"crdnn: 第{use_channel + 1}声道 采样点数{waveform.shape} 采样率{sr}Hz 时长{duration_sec}秒")
        return local_audio_path, duration_sec

    def get_boundaries(self, boundaries_speech, last_ts):
        self.speech_boundaries = []
        self.silence_boundaries = []

        chunk_offset = 0.0
        for i in range(boundaries_speech.shape[0]):
            s_beg = boundaries_speech[i, 0].item()
            s_end = boundaries_speech[i, 1].item()

            if chunk_offset < s_beg:
                self.silence_boundaries.extend([chunk_offset, s_beg])
            self.speech_boundaries.extend([s_beg, s_end])
            chunk_offset = s_end

        if chunk_offset < last_ts:
            self.silence_boundaries.extend([chunk_offset, last_ts])


if __name__ == '__main__':
    parser = ArgumentParser(description="run vad-crdnn-libriparty")
    parser.add_argument('--input', '-i', help='the filepath or URL of audio', type=str)
    parser.add_argument('--output', '-o', help='the filepath of output directory', default='./vad_results', type=str)
    parser.add_argument('--channel', '-c', help='select one channel of audio', default=0, type=int)
    parser.add_argument('--largechunksec', '-l',
                        help='Size (in seconds) of the large chunks that are read sequentially from the input audio file.',
                        default=5, type=float)
    parser.add_argument('--smallchunksec', '-s',
                        help='Size (in seconds) of the small chunks extracted from the large ones. Note that largechunksec/smallchunksec must be an integer.',
                        default=0.5, type=float)
    parser.add_argument('--minadjacentsec', '-m',
                        help='If the silence between speech segments is smaller than parameter, the segments will be merged as speech.',
                        default=0.25, type=float)
    parser.add_argument('--minboundarysec', '-b',
                        help='If the length of the speech segments is smaller than parameter, the segments will be merged as silence.',
                        default=0.25, type=float)
    parser.add_argument('--clean', help='delete history files before run', action="store_true")
    parser.add_argument('--jsonOut', help='save segments as json file or TextGrid file', action="store_true")
    parser.add_argument('--continuous',
                        help='save as continuous or independent(only speech) segments, which applys to --jsonOut ',
                        action="store_true")

    parser.add_argument('--debug', help='profile time complexity', action="store_true")

    args = parser.parse_args()

    vadtask = CRDnnVAD(save_dir=args.output,
                       clean_history=args.clean,
                       save_json=args.jsonOut,
                       save_continuous=args.continuous)
    if args.debug:
        profile.run('vadtask(wav_path=args.input, \
                use_channel=args.channel, \
                large_chunk_sec=args.largechunksec, \
                small_chunk_sec=args.smallchunksec, \
                min_adjacent_sec=args.minadjacentsec, \
                min_boundary_sec=args.minboundarysec)',
                    sort="cumulative")
    else:
        vadtask(wav_path=args.input,
                use_channel=args.channel,
                large_chunk_sec=args.largechunksec,
                small_chunk_sec=args.smallchunksec,
                min_adjacent_sec=args.minadjacentsec,
                min_boundary_sec=args.minboundarysec)
