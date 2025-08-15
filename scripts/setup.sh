#!/bin/bash
set -e

# ffmpeg導入（存在時はスキップ）
if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "[INFO] Installing ffmpeg..."
  if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    sudo apt-get update && sudo apt-get install -y ffmpeg
  elif [[ "$OSTYPE" == "darwin"* ]]; then
    brew install ffmpeg
  else
    echo "[WARN] ffmpegは自動導入されません。手動でインストールしてください。"
  fi
else
  echo "[INFO] ffmpeg already installed."
fi

# external/mangio-rvc clone
if [ ! -d external/mangio-rvc ]; then
  git clone --depth 1 https://github.com/Mangio621/Mangio-RVC-Fork external/mangio-rvc
else
  echo "[INFO] external/mangio-rvc already exists."
fi

# requirements
pip install -r requirements.txt

# 環境チェック
python scripts/env_check.py
