import os
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import webrtcvad
from pydub import AudioSegment
from typing import List
import wave
import contextlib

def convert_to_32k_mono(input_path: str, output_path: str):
    """ffmpegで32kHz/mono変換"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "32000", "-ac", "1", output_path
    ]
    subprocess.run(cmd, check=True)

def trim_silence_vad(input_path: str, output_path: str, aggressiveness: int = 2):
    """webrtcvadで無音トリム"""
    # 音声ファイルを読み込み
    with wave.open(input_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
    
    # VADの初期化
    vad = webrtcvad.Vad(aggressiveness)
    
    # フレームサイズ（30ms）
    frame_size = int(sample_rate * 0.03)
    
    # 音声フレームを分割してVAD処理
    voice_frames = []
    for i in range(0, len(audio_data), frame_size):
        frame = audio_data[i:i + frame_size]
        if len(frame) == frame_size:  # 完全なフレームのみ処理
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)
            if is_speech:
                voice_frames.append(frame)
    
    if not voice_frames:
        print("音声が検出されませんでした")
        return
    
    # 音声フレームを結合
    trimmed_audio = np.concatenate(voice_frames)
    
    # 出力ファイルに保存
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(trimmed_audio.astype(np.int16).tobytes())

def normalize_lufs(input_path: str, output_path: str, target_lufs: float = -23.0):
    """pyloudnormで-23LUFS正規化"""
    # 音声ファイルを読み込み
    audio, sample_rate = sf.read(input_path)
    
    # ステレオの場合はモノラルに変換
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    
    # 現在のLUFSを測定
    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(audio)
    
    # 目標LUFSに正規化
    normalized_audio = pyln.loudness.normalize.loudness(audio, current_lufs, target_lufs)
    
    # 出力ファイルに保存
    sf.write(output_path, normalized_audio, sample_rate)

def split_audio(input_path: str, out_dir: str, chunk_sec: float = 12.0) -> List[str]:
    """約12秒ごとに分割"""
    # 出力ディレクトリを作成
    os.makedirs(out_dir, exist_ok=True)
    
    # 音声ファイルを読み込み
    audio = AudioSegment.from_file(input_path)
    
    # チャンクサイズをミリ秒に変換
    chunk_ms = int(chunk_sec * 1000)
    
    # 音声を分割
    chunks = []
    for i in range(0, len(audio), chunk_ms):
        chunk = audio[i:i + chunk_ms]
        
        # 出力ファイル名を生成
        chunk_filename = f"chunk_{i//chunk_ms:04d}.wav"
        chunk_path = os.path.join(out_dir, chunk_filename)
        
        # チャンクを保存
        chunk.export(chunk_path, format="wav")
        chunks.append(chunk_path)
    
    return chunks

def detect_speech_segments(input_path: str, min_speech_duration: float = 0.5) -> List[tuple]:
    """音声セグメントを検出して時間範囲を返す"""
    # 音声ファイルを読み込み
    with wave.open(input_path, 'rb') as wf:
        sample_rate = wf.getframerate()
        frames = wf.readframes(wf.getnframes())
        audio_data = np.frombuffer(frames, dtype=np.int16)
    
    # VADの初期化
    vad = webrtcvad.Vad(2)  # 中程度のアグレッシブネス
    
    # フレームサイズ（30ms）
    frame_size = int(sample_rate * 0.03)
    
    # 音声フレームを分割してVAD処理
    speech_frames = []
    for i in range(0, len(audio_data), frame_size):
        frame = audio_data[i:i + frame_size]
        if len(frame) == frame_size:
            is_speech = vad.is_speech(frame.tobytes(), sample_rate)
            speech_frames.append(is_speech)
    
    # 連続する音声セグメントを検出
    segments = []
    start_frame = None
    
    for i, is_speech in enumerate(speech_frames):
        if is_speech and start_frame is None:
            start_frame = i
        elif not is_speech and start_frame is not None:
            end_frame = i
            duration = (end_frame - start_frame) * 0.03  # 秒単位
            
            if duration >= min_speech_duration:
                start_time = start_frame * 0.03
                end_time = end_frame * 0.03
                segments.append((start_time, end_time))
            
            start_frame = None
    
    # 最後のセグメントを処理
    if start_frame is not None:
        end_frame = len(speech_frames)
        duration = (end_frame - start_frame) * 0.03
        
        if duration >= min_speech_duration:
            start_time = start_frame * 0.03
            end_time = end_frame * 0.03
            segments.append((start_time, end_time))
    
    return segments

def apply_fade(input_path: str, output_path: str, fade_in_ms: int = 100, fade_out_ms: int = 100):
    """フェードイン・アウトを適用"""
    audio = AudioSegment.from_file(input_path)
    
    # フェードイン・アウトを適用
    audio = audio.fade_in(fade_in_ms).fade_out(fade_out_ms)
    
    # 出力ファイルに保存
    audio.export(output_path, format="wav")

def get_audio_info(input_path: str) -> dict:
    """音声ファイルの情報を取得"""
    try:
        with wave.open(input_path, 'rb') as wf:
            channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            sample_rate = wf.getframerate()
            frames = wf.getnframes()
            duration = frames / sample_rate
        
        return {
            'channels': channels,
            'sample_width': sample_width,
            'sample_rate': sample_rate,
            'frames': frames,
            'duration': duration,
            'format': 'WAV'
        }
    except Exception as e:
        # WAV以外の形式の場合はpydubを使用
        try:
            audio = AudioSegment.from_file(input_path)
            return {
                'channels': audio.channels,
                'sample_width': audio.sample_width,
                'sample_rate': audio.frame_rate,
                'frames': len(audio),
                'duration': len(audio) / 1000.0,  # ミリ秒から秒に変換
                'format': input_path.split('.')[-1].upper()
            }
        except Exception as e2:
            return {'error': f"音声ファイルの読み込みに失敗: {e2}"}
