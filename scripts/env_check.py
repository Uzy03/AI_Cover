import subprocess
import sys

def check_ffmpeg():
    try:
        out = subprocess.check_output(["ffmpeg", "-version"]).decode().split("\n")[0]
        print(f"ffmpeg: {out}")
    except Exception:
        print("ffmpeg: Not found")

def check_cuda():
    try:
        import torch
        print(f"torch: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch: Not installed")
    try:
        import torchaudio
        print(f"torchaudio: {torchaudio.__version__}")
    except ImportError:
        print("torchaudio: Not installed")

def main():
    print("[環境チェック]")
    check_ffmpeg()
    check_cuda()

if __name__ == "__main__":
    main()
