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

## 利用可能なコマンド

### 基本コマンド
- `setup` - セットアップ: 依存導入・外部リポジトリclone・環境チェック
- `download-models` - 事前学習モデルのダウンロード
- `prep` - 音声前処理（32kHz/mono, 無音トリム, LUFS, 分割）
- `train` - 学習プロセスの起動
- `infer` - 推論（音声変換）
- `pack` - モデル一式のパッケージング

### ユーティリティコマンド
- `info` - 音声ファイルの情報を表示
- `config-validate` - 設定ファイルの検証
- `config-create` - 新しい設定ファイルを作成
- `status` - 学習状況の確認
- `extract-features` - 特徴量抽出の実行
- `help` - 利用可能なコマンドの一覧を表示

## 詳細な使用方法

### 1. セットアップ
```bash
# 環境チェックとセットアップ
python -m rvccli setup

# 事前学習モデルのダウンロード
python -m rvccli download-models
```

### 2. 音声前処理
```bash
# 音声ファイルの前処理
python -m rvccli prep --in-dir ./input_audio --out-dir ./processed_audio

# カスタム分割秒数で前処理
python -m rvccli prep --in-dir ./input_audio --out-dir ./processed_audio --chunk-sec 15.0
```

### 3. 学習
```bash
# 学習の実行
python -m rvccli train

# 学習状況の確認
python -m rvccli status
```

### 4. 推論
```bash
# 音声変換
python -m rvccli infer --wav ./input.wav --out ./output.wav

# カスタムパラメータで推論
python -m rvccli infer --wav ./input.wav --out ./output.wav --transpose 2 --f0-method rmvpe
```

### 5. 設定管理
```bash
# 設定ファイルの検証
python -m rvccli config-validate

# 新しい設定ファイルの作成
python -m rvccli config-create --project-name "my_voice_project"
```

### 6. 音声ファイル情報
```bash
# 音声ファイルの詳細情報を表示
python -m rvccli info ./audio_file.wav
```

## 設定ファイル

設定ファイルは `configs/config.yaml` に配置され、以下の設定が可能です：

### 音声設定
- サンプリングレート: 16kHz, 32kHz, 44.1kHz, 48kHz
- チャンネル数: 1（モノラル）, 2（ステレオ）
- LUFS正規化: -70 ～ -10
- VADアグレッシブネス: 0 ～ 3
- チャンク分割秒数: 任意の値
- フェードイン・アウト: ミリ秒単位

### 学習設定
- バッチサイズ: 正の整数
- 学習率: 正の浮動小数点数
- 学習ステップ数: 正の整数
- F0抽出方法: rmvpe, crepe, harvest, pm
- GPU ID: 使用するGPUの番号
- ワーカー数: データローダーのワーカー数

### 推論設定
- 音程シフト: -12 ～ +12
- RMSミックス率: 0 ～ 1
- プロテクト値: 0 ～ 1
- フィルタ半径: 整数値

## 音声処理機能

### 前処理パイプライン
1. **32kHz/mono変換** - FFmpegを使用した高品質な変換
2. **無音トリム** - WebRTC VADによる音声セグメント検出
3. **LUFS正規化** - 標準的な-23LUFSへの正規化
4. **音声分割** - 指定秒数での均等分割
5. **フェード処理** - 自然な音声の開始・終了

### 音声分析機能
- ファイル形式、チャンネル数、サンプリングレートの自動検出
- 音声セグメントの自動検出と時間範囲の特定
- 音声品質の評価とレポート

## エラーハンドリング

- パスの存在確認と自動作成
- 設定ファイルの妥当性検証
- 学習・推論プロセスの詳細なログ出力
- 一時ファイルの自動クリーンアップ
- エラー発生時の適切なメッセージ表示

## パフォーマンス最適化

- FP16（半精度浮動小数点）のサポート
- マルチワーカー処理のサポート
- GPUメモリの効率的な使用
- バッチ処理による高速化

## トラブルシューティング

### よくある問題と解決方法

1. **CUDAエラー**
   ```bash
   # 環境チェック
   python scripts/env_check.py
   
   # PyTorchのCUDA対応確認
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **FFmpegエラー**
   ```bash
   # FFmpegの確認
   which ffmpeg
   
   # セットアップの再実行
   python -m rvccli setup
   ```

3. **設定ファイルエラー**
   ```bash
   # 設定の検証
   python -m rvccli config-validate
   
   # 新しい設定ファイルの作成
   python -m rvccli config-create
   ```

4. **学習の失敗**
   ```bash
   # 学習状況の確認
   python -m rvccli status
   
   # ログファイルの確認
   ls -la outputs/*/
   ```

## 開発者向け情報

### プロジェクト構造
```
rvccli/
├── __init__.py          # パッケージ初期化
├── __main__.py          # メインエントリーポイント
├── audio_utils.py       # 音声処理ユーティリティ
├── cli.py              # CLIコマンド定義
├── config.py           # 設定管理クラス
├── download_models.py  # モデルダウンロード
└── rvc_wrapper.py      # RVCスクリプトラッパー
```

### 依存関係
- **音声処理**: soundfile, pyloudnorm, webrtcvad, pydub
- **数値計算**: numpy
- **CLI**: typer
- **設定**: pyyaml
- **外部ツール**: ffmpeg

### 拡張方法
- 新しい音声処理アルゴリズムの追加
- カスタム設定パラメータの追加
- 新しいCLIコマンドの実装
- バッチ処理の最適化

## ライセンスと貢献

本プロジェクトはMITライセンスの下で公開されています。バグ報告、機能要求、プルリクエストを歓迎します。

### 貢献方法
1. このリポジトリをフォーク
2. 機能ブランチを作成
3. 変更をコミット
4. プルリクエストを作成

## 更新履歴

- **v1.0.0** - 初期リリース、基本的なRVCパイプライン
- **v1.1.0** - 音声処理機能の強化、設定管理の改善
- **v1.2.0** - エラーハンドリングの強化、新しいCLIコマンドの追加
