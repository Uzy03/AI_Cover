import os
import subprocess
import numpy as np
import soundfile as sf
import pyloudnorm as pyln
import webrtcvad
from pydub import AudioSegment
from typing import List

def convert_to_32k_mono(input_path: str, output_path: str):
    """ffmpegで32kHz/mono変換"""
    cmd = [
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "32000", "-ac", "1", output_path
    ]
    subprocess.run(cmd, check=True)

def trim_silence_vad(input_path: str, output_path: str, aggressiveness: int = 2):
    """webrtcvadで無音トリム"""
    # 実装は省略（雛形）
    pass

def normalize_lufs(input_path: str, output_path: str, target_lufs: float = -23.0):
    """pyloudnormで-23LUFS正規化"""
    # 実装は省略（雛形）
    pass

def split_audio(input_path: str, out_dir: str, chunk_sec: float = 12.0) -> List[str]:
    """約12秒ごとに分割"""
    # 実装は省略（雛形）
    return []
