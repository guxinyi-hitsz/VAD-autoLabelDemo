# -*- coding: UTF-8 -*- 

"""
Author  : Xinyi Gu
Date    : 2023/3/17 16:27 
Project :
Description:

WebrtcVad 的并行版本，处理长音频

Change History: 
    @change
"""
import itertools
import profile
from abstract import VADBase
from librosa import resample
import numpy as np
from webrtcvad import Vad, valid_rate_and_frame_length
from malaya_speech import Pipeline
from malaya_speech.utils import generator
from malaya_speech.extra.visualization import visualize_vad
from malaya_speech import vad
from malaya_speech.model.webrtc import WebRTC
from pathlib import Path
from argparse import ArgumentParser

__all__ = ['GMMVAD']


class BatchWebRTC(WebRTC):
    __name__ = 'this is a wrapper class for webrtcvad, support a list of Frames as input'

    def __str__(self):
        return f'<{self.__name__}>'

    def __init__(self, vad, sample_rate=16000, minimum_amplitude: int = 100):
        super().__init__(vad, sample_rate, minimum_amplitude)

    def __call__(self, frameList):
        return self.is_speech(frameList)

    def is_speech(self, frameList):
        """override"""
        for frame in frameList:
            assert valid_rate_and_frame_length(self.sample_rate, frame.array.shape[
                0]), f'webrtcvad 仅支持[10ms,30ms]的音频块长度 而实际输入{int(frame.array.shape[0] / self.sample_rate * 1000)}ms'
        return [super(BatchWebRTC, self).is_speech(frame) for frame in frameList]


class GMMVAD(VADBase):
    def __init__(self,
                 save_dir='./vad_results',
                 audio_dir='./vad_cache',
                 clean_history=False,
                 save_json=False,
                 save_continuous=False,
                 use_parallel=True
                 ):
        super().__init__(save_dir, audio_dir, clean_history, save_json, save_continuous)
        self.use_parallel = use_parallel

    def prepare_audio(self, wav_path, use_channel=0):
        """override
        The WebRTC VAD only accepts 16-bit mono PCM audio, sampled at 8000, 16000, 32000 or 48000 Hz.
        """
        wav_vector, sr = super().prepare_audio(wav_path, use_channel)

        if sr not in [8000, 16000, 32000, 48000]:
            wav_vector = resample(wav_vector, orig_sr=sr, target_sr=16000)
            sr = 16000

        if wav_vector.dtype != np.int16:
            if np.max(np.abs(wav_vector)) == 0:
                wav_vector = np.zeros(wav_vector.shape, dtype=np.int16)
            else:
                normalized_vector = wav_vector / (np.max(np.abs(wav_vector)) + 1e-9)
                scaled_vector = normalized_vector * np.iinfo(np.int16).max
                wav_vector = scaled_vector.astype(np.int16)

        duration_sec = wav_vector.shape[0] / sr

        print(f"webrtc: 第{use_channel + 1}声道 采样点数{wav_vector.shape} 采样率{sr}Hz 时长{duration_sec}秒")

        return wav_vector, sr, duration_sec

    def __call__(self,
                 wav_path,
                 use_channel=0,
                 large_chunk_sec=5,
                 small_chunk_sec=0.02,
                 min_adjacent_sec=0.25,
                 min_boundary_sec=0.25,
                 aggressive_mode=3,
                 check_loop=3
                 ):
        wav_vector, sr, duration_sec = self.prepare_audio(wav_path, use_channel)  # 5% 耗时
        chunk_ms = int(small_chunk_sec * 1000)

        # Split audio samples into small chunks, each small_chunk has 3 attributes: array(np.ndarray, dtype=float64), duration(float), timestamp(float)
        def malaya_wrapper(frame_duration_ms, sample_rate):
            def inner(audio_ndarray):
                return generator.frames(audio=audio_ndarray,
                                        frame_duration_ms=frame_duration_ms,
                                        sample_rate=sample_rate,
                                        append_ending_trail=False)

            return inner

        new_splitter = malaya_wrapper(frame_duration_ms=chunk_ms, sample_rate=sr)

        # 建Pipeline
        if self.use_parallel:
            # 开启批处理
            batchsize = int(large_chunk_sec / small_chunk_sec)
            model_gmmvad = BatchWebRTC(vad=Vad(mode=aggressive_mode),
                                       sample_rate=sr,
                                       minimum_amplitude=int(np.quantile(np.abs(wav_vector), 0.05))
                                       )

            p = Pipeline()
            chunks = (
                p.map(new_splitter)
                .batching(batchsize)
            )
            vad_map = chunks.foreach_map(model_gmmvad, method='thread')
            zip_map = chunks.flatten().foreach_zip(vad_map.flatten())
        else:
            # 单线程
            model_gmmvad = vad.webrtc(aggressiveness=aggressive_mode,
                                      sample_rate=sr,
                                      minimum_amplitude=int(np.quantile(np.abs(wav_vector), 0.05))
                                      )

            p = Pipeline()
            chunks = (
                p.map(new_splitter)
            )
            vad_map = chunks.foreach_map(model_gmmvad)
            zip_map = chunks.foreach_zip(vad_map)

        # 运行
        batch_result = p.emit(wav_vector)
        vad_chunks = list(batch_result['foreach_zip'])

        # 可视化
        # visualize_vad(signal=wav_vector, preds=vad_chunks, sample_rate=sr)

        # 合并无效分段
        self.after_process(preds=vad_chunks, close_th=min_adjacent_sec, len_th=min_boundary_sec, check_loop=check_loop)

        # 导出vad结果
        flags = [is_speech_flag for chunk, is_speech_flag in vad_chunks]
        self.speech_boundaries = []
        self.silence_boundaries = []
        chunk_offset = 0
        for flagType, flagList in itertools.groupby(flags):
            prev_offset = chunk_offset
            chunk_offset += len(list(flagList))
            start_sec = vad_chunks[prev_offset][0].timestamp
            end_sec = vad_chunks[chunk_offset - 1][0].timestamp + vad_chunks[chunk_offset - 1][0].duration
            if chunk_offset == len(flags):
                # append ending trail
                end_sec = duration_sec

            if flagType:
                self.speech_boundaries.extend([start_sec, end_sec])
            else:
                self.silence_boundaries.extend([start_sec, end_sec])
        # 导出到文件
        self.save_boundaries(filename=str(Path(wav_path).stem) + self.option["save_format"])

    @staticmethod
    def after_process(preds, close_th, len_th, check_loop):
        check_loop = max(check_loop, 3)
        for loop in range(check_loop):
            flags = [is_speech_flag for chunk, is_speech_flag in preds]
            chunk_offset = 0
            for flagType, flagList in itertools.groupby(flags):
                prev_offset = chunk_offset
                chunk_offset += len(list(flagList))
                start_sec = preds[prev_offset][0].timestamp
                end_sec = preds[chunk_offset - 1][0].timestamp + preds[chunk_offset - 1][0].duration
                # Merge short speech segments
                if not flagType and (end_sec - start_sec) <= close_th:
                    for i in range(prev_offset, chunk_offset):
                        preds[i] = (preds[i][0], True)

                # Remove short speech segments
                if loop == (check_loop - 1) and flagType and (end_sec - start_sec) <= len_th:
                    for i in range(prev_offset, chunk_offset):
                        preds[i] = (preds[i][0], False)


if __name__ == '__main__':
    parser = ArgumentParser(description="run vad-webrtc")

    parser.add_argument('--input', '-i', help='the filepath or URL of audio', type=str)
    parser.add_argument('--output', '-o', help='the filepath of output directory', default='./vad_results', type=str)
    parser.add_argument('--channel', '-c', help='select one channel of audio', default=0, type=int)
    parser.add_argument('--largechunksec', '-l',
                        help='Size (in seconds) of the large chunks that are processed as batches.',
                        default=5, type=float)
    parser.add_argument('--smallchunksec', '-s',
                        help='Size (in seconds) of the small chunks, must be in range [0.01, 0.03] sec.',
                        default=0.02, type=float)
    parser.add_argument('--minadjacentsec', '-m',
                        help='If the silence between speech segments is smaller than parameter, the segments will be merged as speech.',
                        default=1.0, type=float)
    parser.add_argument('--minboundarysec', '-b',
                        help='If the length of the speech segments is smaller than parameter, the segments will be merged as silence.',
                        default=0.5, type=float)
    parser.add_argument('--clean', help='delete history files before run', action="store_true")
    parser.add_argument('--jsonOut', help='save segments as json file or TextGrid file', action="store_true")
    parser.add_argument('--continuous', help='save as continuous or independent(only speech) segments, which applys to --jsonOut ', action="store_true")

    parser.add_argument('--mode', help='vad mode choices: [0, 1, 2, 3] Larger number, the more probability detected as silence', default=3, type=int)
    parser.add_argument('--loop', help='after process loop times. Recommend at least enough times to make satisfied merged segments.', default=15, type=int)

    parser.add_argument('--debug', help='profile time complexity', action="store_true")

    args = parser.parse_args()
    vadtask = GMMVAD(save_dir=args.output,
                     clean_history=args.clean,
                     save_json=args.jsonOut,
                     save_continuous=args.continuous,
                     use_parallel=True)
    if args.debug:
        profile.run('vadtask(wav_path=args.input, \
                    use_channel=args.channel, \
                    large_chunk_sec=args.largechunksec, \
                    small_chunk_sec=args.smallchunksec, \
                    min_adjacent_sec=args.minadjacentsec, \
                    min_boundary_sec=args.minboundarysec, \
                    aggressive_mode=args.mode, \
                    check_loop=args.loop)',
                    sort="cumulative")
    else:
        vadtask(wav_path=args.input,
                use_channel=args.channel,
                large_chunk_sec=args.largechunksec,
                small_chunk_sec=args.smallchunksec,
                min_adjacent_sec=args.minadjacentsec,
                min_boundary_sec=args.minboundarysec,
                aggressive_mode=args.mode,
                check_loop=args.loop)
