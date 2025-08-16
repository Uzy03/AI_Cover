import os
import requests

def download_contentvec(models_dir: str):
    """ContentVecモデルのダウンロード"""
    url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/contentvec/pretrained/contentvec.pth"
    filename = "contentvec.pth"
    filepath = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading ContentVec model to {filepath}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("ContentVec model downloaded.")
        except Exception as e:
            print(f"Failed to download ContentVec: {e}")
    else:
        print("ContentVec model already exists.")

def download_rmvpe(models_dir: str):
    """RMVPEモデルのダウンロード"""
    url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/rmvpe.pt"
    filename = "rmvpe.pt"
    filepath = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading RMVPE model to {filepath}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("RMVPE model downloaded.")
        except Exception as e:
            print(f"Failed to download RMVPE: {e}")
    else:
        print("RMVPE model already exists.")

def download_crepe(models_dir: str):
    """CREPEモデルのダウンロード"""
    url = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main/crepe_onnx_full.onnx"
    filename = "crepe_onnx_full.onnx"
    filepath = os.path.join(models_dir, filename)
    if not os.path.exists(filepath):
        print(f"Downloading CREPE model to {filepath}...")
        try:
            r = requests.get(url, stream=True)
            r.raise_for_status()
            with open(filepath, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("CREPE model downloaded.")
        except Exception as e:
            print(f"Failed to download CREPE: {e}")
    else:
        print("CREPE model already exists.")

def ensure_models(models_dir: str):
    """必要モデルの存在確認とDL/手動配置案内"""
    os.makedirs(models_dir, exist_ok=True)
    print("Checking and downloading required models...")
    download_contentvec(models_dir)
    download_rmvpe(models_dir)
    download_crepe(models_dir)
    print("All required models are ready.")
