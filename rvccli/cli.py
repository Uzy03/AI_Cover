import typer
import os
import shutil
import subprocess
from pathlib import Path

app = typer.Typer(help="Retrieval-based Voice Conversion CLI")

@app.command()
def help():
    """利用可能なコマンドの一覧を表示"""
    print("RVC CLI - 利用可能なコマンド")
    print("=" * 50)
    print()
    
    commands = [
        ("setup", "セットアップ: 依存導入・外部リポジトリclone・環境チェック"),
        ("download-models", "事前学習モデルのダウンロード"),
        ("prep", "音声前処理（32kHz/mono, 無音トリム, LUFS, 分割）"),
        ("train", "学習プロセスの起動"),
        ("infer", "推論（音声変換）"),
        ("pack", "モデル一式のパッケージング"),
        ("info", "音声ファイルの情報を表示"),
        ("config-validate", "設定ファイルの検証"),
        ("config-create", "新しい設定ファイルを作成"),
        ("status", "学習状況の確認"),
        ("extract-features", "特徴量抽出の実行"),
        ("help", "このヘルプを表示")
    ]
    
    for cmd, desc in commands:
        print(f"{cmd:<20} {desc}")
    
    print()
    print("詳細なヘルプは以下のコマンドで確認できます:")
    print("  python -m rvccli <コマンド名> --help")
    print()
    print("例:")
    print("  python -m rvccli prep --help")
    print("  python -m rvccli train --help")

@app.command()
def setup():
    """セットアップ: 依存導入・外部リポジトリclone・環境チェック"""
    print("RVC CLIのセットアップを開始します...")
    
    # 必要なディレクトリを作成
    dirs_to_create = ['models', 'data', 'data/chunks', 'outputs', 'temp']
    for dir_name in dirs_to_create:
        dir_path = os.path.join(os.path.dirname(__file__), '..', dir_name)
        os.makedirs(dir_path, exist_ok=True)
        print(f"ディレクトリを作成/確認: {dir_path}")
    
    # 設定ファイルの確認
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    example_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.example.yaml')
    
    if not os.path.exists(config_path) and os.path.exists(example_path):
        shutil.copy(example_path, config_path)
        print(f"設定ファイルを作成: {config_path}")
    
    # 環境チェック
    print("\n環境チェックを実行中...")
    try:
        import numpy
        print("✓ NumPy: OK")
    except ImportError:
        print("✗ NumPy: インストールが必要")
    
    try:
        import soundfile
        print("✓ SoundFile: OK")
    except ImportError:
        print("✗ SoundFile: インストールが必要")
    
    try:
        import pyloudnorm
        print("✓ PyLoudNorm: OK")
    except ImportError:
        print("✗ PyLoudNorm: インストールが必要")
    
    try:
        import webrtcvad
        print("✓ WebRTC VAD: OK")
    except ImportError:
        print("✗ WebRTC VAD: インストールが必要")
    
    try:
        import pydub
        print("✓ PyDub: OK")
    except ImportError:
        print("✗ PyDub: インストールが必要")
    
    # ffmpegの確認
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
        if result.returncode == 0:
            print("✓ FFmpeg: OK")
        else:
            print("✗ FFmpeg: インストールが必要")
    except FileNotFoundError:
        print("✗ FFmpeg: インストールが必要")
    
    print("\nセットアップが完了しました。")
    print("必要な依存関係がインストールされていない場合は、以下のコマンドでインストールしてください:")
    print("pip install -r requirements.txt")

@app.command("download-models")
def download_models():
    """事前学習モデルのダウンロード"""
    import os
    from . import download_models as dm
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models_dir = os.path.abspath(models_dir)
    dm.ensure_models(models_dir)

@app.command()
def prep(in_dir: str = typer.Option(..., help="入力ディレクトリ"), 
         out_dir: str = typer.Option(..., help="出力ディレクトリ"),
         chunk_sec: float = typer.Option(12.0, help="分割秒数")):
    """音声前処理（32kHz/mono, 無音トリム, LUFS, 分割）"""
    from . import audio_utils
    import glob
    
    print(f"音声前処理を開始します...")
    print(f"入力ディレクトリ: {in_dir}")
    print(f"出力ディレクトリ: {out_dir}")
    
    # 出力ディレクトリを作成
    os.makedirs(out_dir, exist_ok=True)
    
    # 音声ファイルを検索
    audio_extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
    audio_files = []
    for ext in audio_extensions:
        audio_files.extend(glob.glob(os.path.join(in_dir, ext)))
        audio_files.extend(glob.glob(os.path.join(in_dir, ext.upper())))
    
    if not audio_files:
        print("音声ファイルが見つかりませんでした。")
        return
    
    print(f"処理対象ファイル数: {len(audio_files)}")
    
    for i, audio_file in enumerate(audio_files, 1):
        print(f"\n処理中 ({i}/{len(audio_files)}): {os.path.basename(audio_file)}")
        
        # 一時ファイル名を生成
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        temp_32k = os.path.join(out_dir, f"{base_name}_32k.wav")
        temp_trimmed = os.path.join(out_dir, f"{base_name}_trimmed.wav")
        temp_normalized = os.path.join(out_dir, f"{base_name}_normalized.wav")
        
        try:
            # 1. 32kHz/mono変換
            print("  32kHz/mono変換中...")
            audio_utils.convert_to_32k_mono(audio_file, temp_32k)
            
            # 2. 無音トリム
            print("  無音トリム中...")
            audio_utils.trim_silence_vad(temp_32k, temp_trimmed)
            
            # 3. LUFS正規化
            print("  LUFS正規化中...")
            audio_utils.normalize_lufs(temp_trimmed, temp_normalized)
            
            # 4. 音声分割
            print("  音声分割中...")
            chunks_dir = os.path.join(out_dir, f"{base_name}_chunks")
            chunks = audio_utils.split_audio(temp_normalized, chunks_dir, chunk_sec)
            
            print(f"  分割完了: {len(chunks)}個のチャンク")
            
            # 一時ファイルを削除
            for temp_file in [temp_32k, temp_trimmed, temp_normalized]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            
        except Exception as e:
            print(f"  エラー: {e}")
            continue
    
    print(f"\n音声前処理が完了しました。出力ディレクトリ: {out_dir}")

@app.command()
def train():
    """学習プロセスの起動"""
    import os
    from . import config, rvc_wrapper
    # 設定ファイルのパス（なければexampleをコピー）
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    if not os.path.exists(config_path):
        example_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.example.yaml')
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        import shutil
        shutil.copy(example_path, config_path)
        print(f"config.yamlが見つからないため、exampleからコピーしました: {config_path}")
    
    try:
        cfg = config.RVCConfig.load(config_path)
        
        # 設定の検証
        errors = cfg.validate()
        if errors:
            print("設定ファイルに問題があります:")
            for error in errors:
                print(f"  ✗ {error}")
            return
        
        print("設定ファイルの検証が完了しました")
        
        # データセットディレクトリ（仮）
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks'))
        out_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'outputs'))
        
        print("学習を開始します...")
        success = rvc_wrapper.train(
            dataset_dir=dataset_dir,
            sr=cfg.sr,
            f0_method=cfg.f0_method,
            batch=cfg.batch,
            steps=cfg.steps,
            fp16=cfg.fp16,
            out_dir=out_dir,
            index_rate=cfg.training.index_rate,
            save_every_n=cfg.training.save_every_n
        )
        
        if success:
            print("学習が完了しました。")
        else:
            print("学習に失敗しました。")
            
    except Exception as e:
        print(f"設定ファイルの読み込みまたは学習の実行に失敗しました: {e}")

@app.command()
def infer(wav: str = typer.Option(..., help="入力wav"), 
          out: str = typer.Option(..., help="出力wav"),
          model_path: str = typer.Option(None, help="モデルパス"),
          index_path: str = typer.Option(None, help="インデックスパス"),
          transpose: int = typer.Option(0, help="音程シフト"),
          f0_method: str = typer.Option("rmvpe", help="F0抽出方法")):
    """推論（音声変換）"""
    from . import rvc_wrapper
    import os
    
    print(f"音声変換を開始します...")
    print(f"入力ファイル: {wav}")
    print(f"出力ファイル: {out}")
    
    # モデルパスとインデックスパスの自動検出
    if model_path is None:
        # デフォルトのモデルディレクトリから検索
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.pth')]
        if model_files:
            model_path = os.path.join(models_dir, model_files[0])
            print(f"自動検出されたモデル: {model_path}")
        else:
            print("エラー: モデルファイルが見つかりません。--model-pathで指定してください。")
            return
    
    if index_path is None:
        # デフォルトのインデックスディレクトリから検索
        models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        index_files = [f for f in os.listdir(models_dir) if f.endswith('.index')]
        if index_files:
            index_path = os.path.join(models_dir, index_files[0])
            print(f"自動検出されたインデックス: {index_path}")
        else:
            print("エラー: インデックスファイルが見つかりません。--index-pathで指定してください。")
            return
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(out), exist_ok=True)
    
    try:
        success = rvc_wrapper.infer(
            input_wav=wav,
            model_path=model_path,
            index_path=index_path,
            transpose=transpose,
            f0_method=f0_method,
            rms_mix_rate=0.25,
            filter_radius=3,
            resample_sr=0,
            out_path=out
        )
        
        if success:
            print(f"音声変換が完了しました: {out}")
        else:
            print("音声変換に失敗しました。")
            
    except Exception as e:
        print(f"音声変換でエラーが発生しました: {e}")

@app.command()
def pack():
    """モデル一式のパッケージング"""
    import os
    import shutil
    import zipfile
    from datetime import datetime
    
    print("モデルのパッケージングを開始します...")
    
    # 出力ディレクトリ
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    if not os.path.exists(outputs_dir):
        print("エラー: 出力ディレクトリが見つかりません。先に学習を実行してください。")
        return
    
    # 最新の学習結果を検索
    output_subdirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    if not output_subdirs:
        print("エラー: 学習結果が見つかりません。先に学習を実行してください。")
        return
    
    # 最新のディレクトリを選択（作成日時でソート）
    latest_dir = max(output_subdirs, key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)))
    latest_path = os.path.join(outputs_dir, latest_dir)
    
    print(f"パッケージング対象: {latest_dir}")
    
    # パッケージ名を生成
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"rvc_model_{latest_dir}_{timestamp}.zip"
    package_path = os.path.join(outputs_dir, package_name)
    
    # ZIPファイルを作成
    with zipfile.ZipFile(package_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # 学習結果を追加
        for root, dirs, files in os.walk(latest_path):
            for file in files:
                file_path = os.path.join(root, file)
                arc_name = os.path.relpath(file_path, outputs_dir)
                zipf.write(file_path, arc_name)
                print(f"  追加: {arc_name}")
        
        # 必要なモデルファイルを追加
        required_models = ['contentvec.pth', 'rmvpe.pt', 'crepe_onnx_full.onnx']
        for model in required_models:
            model_path = os.path.join(models_dir, model)
            if os.path.exists(model_path):
                zipf.write(model_path, f"models/{model}")
                print(f"  追加: models/{model}")
    
    print(f"\nパッケージングが完了しました: {package_path}")
    print(f"ファイルサイズ: {os.path.getsize(package_path) / (1024*1024):.1f} MB")

@app.command()
def info(wav: str = typer.Option(..., help="音声ファイルパス")):
    """音声ファイルの情報を表示"""
    from . import audio_utils
    
    if not os.path.exists(wav):
        print(f"エラー: ファイルが見つかりません: {wav}")
        return
    
    print(f"音声ファイル情報: {wav}")
    print("-" * 50)
    
    try:
        info = audio_utils.get_audio_info(wav)
        if 'error' in info:
            print(f"エラー: {info['error']}")
            return
        
        print(f"形式: {info['format']}")
        print(f"チャンネル数: {info['channels']}")
        print(f"サンプル幅: {info['sample_width']} bytes")
        print(f"サンプリングレート: {info['sample_rate']:,} Hz")
        print(f"フレーム数: {info['frames']:,}")
        print(f"長さ: {info['duration']:.2f} 秒")
        
        # 音声セグメントの検出
        print("\n音声セグメント検出中...")
        segments = audio_utils.detect_speech_segments(wav)
        if segments:
            print(f"検出された音声セグメント数: {len(segments)}")
            for i, (start, end) in enumerate(segments[:5], 1):  # 最初の5個のみ表示
                print(f"  セグメント {i}: {start:.2f}s - {end:.2f}s (長さ: {end-start:.2f}s)")
            if len(segments) > 5:
                print(f"  ... 他 {len(segments)-5} 個のセグメント")
        else:
            print("音声セグメントが検出されませんでした")
            
    except Exception as e:
        print(f"エラー: {e}")

@app.command()
def config_validate():
    """設定ファイルの検証"""
    import os
    from . import config
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    
    if not os.path.exists(config_path):
        print(f"エラー: 設定ファイルが見つかりません: {config_path}")
        return
    
    try:
        cfg = config.RVCConfig.load(config_path)
        print("設定ファイルの読み込みが完了しました")
        
        # 設定の検証
        errors = cfg.validate()
        if errors:
            print("\n設定ファイルに問題があります:")
            for error in errors:
                print(f"  ✗ {error}")
            return False
        else:
            print("✓ 設定ファイルの検証が完了しました")
            return True
            
    except Exception as e:
        print(f"設定ファイルの検証に失敗しました: {e}")
        return False

@app.command()
def config_create(project_name: str = typer.Option("my_voice_clone", help="プロジェクト名")):
    """新しい設定ファイルを作成"""
    import os
    from . import config
    
    config_path = os.path.join(os.path.dirname(__file__), '..', 'configs', 'config.yaml')
    
    if os.path.exists(config_path):
        print(f"設定ファイルは既に存在します: {config_path}")
        overwrite = input("上書きしますか？ (y/N): ")
        if overwrite.lower() != 'y':
            print("設定ファイルの作成をキャンセルしました")
            return
    
    try:
        # デフォルト設定を作成
        cfg = config.RVCConfig.create_default(project_name)
        
        # 設定ファイルを保存
        cfg.save(config_path)
        print(f"新しい設定ファイルを作成しました: {config_path}")
        print(f"プロジェクト名: {cfg.project_name}")
        print(f"モデル名: {cfg.model_name}")
        
    except Exception as e:
        print(f"設定ファイルの作成に失敗しました: {e}")

@app.command()
def status():
    """学習状況の確認"""
    import os
    from . import rvc_wrapper
    
    outputs_dir = os.path.join(os.path.dirname(__file__), '..', 'outputs')
    
    if not os.path.exists(outputs_dir):
        print("出力ディレクトリが存在しません。先に学習を実行してください。")
        return
    
    # 最新の学習結果を検索
    output_subdirs = [d for d in os.listdir(outputs_dir) if os.path.isdir(os.path.join(outputs_dir, d))]
    if not output_subdirs:
        print("学習結果が見つかりません。先に学習を実行してください。")
        return
    
    # 最新のディレクトリを選択
    latest_dir = max(output_subdirs, key=lambda x: os.path.getctime(os.path.join(outputs_dir, x)))
    latest_path = os.path.join(outputs_dir, latest_dir)
    
    print(f"最新の学習結果: {latest_dir}")
    print(f"パス: {latest_path}")
    
    # 学習状況を確認
    rvc_wrapper.get_training_status(latest_path)

@app.command()
def extract_features(dataset_dir: str = typer.Option(None, help="データセットディレクトリ")):
    """特徴量抽出の実行"""
    import os
    from . import rvc_wrapper, config
    
    if dataset_dir is None:
        # デフォルトのデータセットディレクトリ
        dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', 'chunks'))
    
    if not os.path.exists(dataset_dir):
        print(f"エラー: データセットディレクトリが見つかりません: {dataset_dir}")
        return
    
    print(f"特徴量抽出を開始します...")
    print(f"データセットディレクトリ: {dataset_dir}")
    
    try:
        success = rvc_wrapper.extract_features(dataset_dir)
        if success:
            print("特徴量抽出が完了しました")
        else:
            print("特徴量抽出に失敗しました")
            
    except Exception as e:
        print(f"特徴量抽出でエラーが発生しました: {e}")

if __name__ == "__main__":
    app()
