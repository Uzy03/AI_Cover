import os
import subprocess

def train(dataset_dir, sr, f0_method, batch, steps, fp16, out_dir, index_rate, save_every_n):
    """Mangio-RVC-Forkの学習スクリプトを呼び出し"""
    # 実装は省略（雛形）
    pass

def infer(input_wav, model_path, index_path, transpose, f0_method, rms_mix_rate, filter_radius, resample_sr, out_path):
    """Mangio-RVC-Forkの推論スクリプトを呼び出し"""
    # 実装は省略（雛形）
    pass
