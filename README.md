# rvccli: Retrieval-based Voice Conversion CLI

## 概要
本ツールは [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) をCLIから簡単に利用できる音声変換（RVC）学習・推論パイプラインです。WebUIは使わず、Linux/Colab/RunPod等のCLI環境で動作します。

- 入力音声の前処理（32kHz/mono, 無音トリム, -23LUFS正規化, 12秒分割）
- ContentVec/RMVPE/CREPE等の特徴量・モデル自動DL
- Mangio-RVC-Forkの学習・推論スクリプトを安全にラップ
- Typer製CLI: `rvccli`

## 法的注意
- 声の権利者の許諾なく第三者の声を学習・公開・配布することは禁止されています。
- 本ツールの利用は自己責任で行ってください。

## セットアップ手順
1. CUDA対応PyTorchをインストール（例: CUDA 12.1）
   ```sh
   pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```
2. 依存・外部リポジトリ導入
   ```sh
   make setup
   ```
3. 事前学習モデルDL
   ```sh
   make download
   ```
4. データ前処理
   ```sh
   make prep IN=./samples
   ```
5. 学習
   ```sh
   make train
   ```
6. 推論
   ```sh
   make infer WAV=./test.wav OUT=./outputs/converted.wav
   ```

## 典型値
- サンプリングレート: 32kHz
- f0推定: RMVPE
- バッチ: 4
- ステップ: 20,000
- T4で1–2時間目安

## トラブルシュート
- CUDAが認識されない: `python scripts/env_check.py` で確認。PyTorch, CUDA, ドライバを再確認。
- ffmpegが見つからない: `which ffmpeg` で確認。`make setup` で自動導入。
- モデルDL失敗: `rvccli download-models` 実行。失敗時は`models/`に手動配置。

## ライセンス
MIT License
