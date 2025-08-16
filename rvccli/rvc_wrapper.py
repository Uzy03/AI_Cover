import os
import subprocess
import logging
from pathlib import Path

# ログ設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def _check_rvc_repository():
    """Mangio-RVC-Forkリポジトリの存在確認"""
    rvc_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Mangio-RVC-Fork'))
    if not os.path.exists(rvc_dir):
        logger.warning(f"Mangio-RVC-Forkリポジトリが見つかりません: {rvc_dir}")
        logger.info("以下のコマンドでクローンしてください:")
        logger.info("git clone https://github.com/Mangio621/Mangio-RVC-Fork.git")
        return False
    return True

def _validate_paths(*paths):
    """パスの存在確認"""
    for path in paths:
        if path and not os.path.exists(path):
            raise FileNotFoundError(f"パスが見つかりません: {path}")

def train(dataset_dir, sr, f0_method, batch, steps, fp16, out_dir, index_rate, save_every_n):
    """Mangio-RVC-Forkの学習スクリプトを呼び出し"""
    logger.info("学習プロセスを開始します...")
    
    # パスの検証
    try:
        _validate_paths(dataset_dir)
    except FileNotFoundError as e:
        logger.error(f"データセットディレクトリの検証に失敗: {e}")
        return False
    
    # RVCリポジトリの確認
    if not _check_rvc_repository():
        return False
    
    # 学習スクリプトのパス
    train_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Mangio-RVC-Fork', 'train.py'))
    if not os.path.exists(train_script):
        logger.error(f"学習スクリプトが見つかりません: {train_script}")
        return False
    
    # 出力ディレクトリの作成
    os.makedirs(out_dir, exist_ok=True)
    
    # 学習コマンドの構築
    cmd = [
        'python', train_script,
        '--dataset', dataset_dir,
        '--sr', str(sr),
        '--f0_method', f0_method,
        '--batch', str(batch),
        '--steps', str(steps),
        '--fp16', str(fp16),
        '--out', out_dir,
        '--index_rate', str(index_rate),
        '--save_every_n', str(save_every_n)
    ]
    
    logger.info(f"実行コマンド: {' '.join(cmd)}")
    logger.info(f"データセット: {dataset_dir}")
    logger.info(f"出力ディレクトリ: {out_dir}")
    logger.info(f"サンプリングレート: {sr}Hz")
    logger.info(f"F0抽出方法: {f0_method}")
    logger.info(f"バッチサイズ: {batch}")
    logger.info(f"学習ステップ数: {steps}")
    
    try:
        # 学習プロセスの実行
        logger.info("学習スクリプトを実行中...")
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=False,  # リアルタイムでログを表示
            text=True
        )
        logger.info("学習が正常に完了しました")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"学習スクリプトの実行に失敗しました: {e}")
        logger.error(f"終了コード: {e.returncode}")
        return False
        
    except KeyboardInterrupt:
        logger.warning("学習がユーザーによって中断されました")
        return False
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return False

def infer(input_wav, model_path, index_path, transpose, f0_method, rms_mix_rate, filter_radius, resample_sr, out_path):
    """Mangio-RVC-Forkの推論スクリプトを呼び出し"""
    logger.info("推論プロセスを開始します...")
    
    # パスの検証
    try:
        _validate_paths(input_wav, model_path, index_path)
    except FileNotFoundError as e:
        logger.error(f"入力ファイルの検証に失敗: {e}")
        return False
    
    # RVCリポジトリの確認
    if not _check_rvc_repository():
        return False
    
    # 推論スクリプトのパス
    infer_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Mangio-RVC-Fork', 'infer.py'))
    if not os.path.exists(infer_script):
        logger.error(f"推論スクリプトが見つかりません: {infer_script}")
        return False
    
    # 出力ディレクトリの作成
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    
    # 推論コマンドの構築
    cmd = [
        'python', infer_script,
        '--input', input_wav,
        '--model', model_path,
        '--index', index_path,
        '--transpose', str(transpose),
        '--f0_method', f0_method,
        '--rms_mix_rate', str(rms_mix_rate),
        '--filter_radius', str(filter_radius),
        '--resample_sr', str(resample_sr),
        '--out', out_path
    ]
    
    logger.info(f"実行コマンド: {' '.join(cmd)}")
    logger.info(f"入力ファイル: {input_wav}")
    logger.info(f"モデルファイル: {model_path}")
    logger.info(f"インデックスファイル: {index_path}")
    logger.info(f"出力ファイル: {out_path}")
    logger.info(f"音程シフト: {transpose}")
    logger.info(f"F0抽出方法: {f0_method}")
    
    try:
        # 推論プロセスの実行
        logger.info("推論スクリプトを実行中...")
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=False,  # リアルタイムでログを表示
            text=True
        )
        logger.info("推論が正常に完了しました")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"推論スクリプトの実行に失敗しました: {e}")
        logger.error(f"終了コード: {e.returncode}")
        return False
        
    except KeyboardInterrupt:
        logger.warning("推論がユーザーによって中断されました")
        return False
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return False

def extract_features(dataset_dir, f0_method="rmvpe"):
    """特徴量抽出プロセス"""
    logger.info("特徴量抽出を開始します...")
    
    # パスの検証
    try:
        _validate_paths(dataset_dir)
    except FileNotFoundError as e:
        logger.error(f"データセットディレクトリの検証に失敗: {e}")
        return False
    
    # RVCリポジトリの確認
    if not _check_rvc_repository():
        return False
    
    # 特徴量抽出スクリプトのパス
    extract_script = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Mangio-RVC-Fork', 'extract_feature.py'))
    if not os.path.exists(extract_script):
        logger.error(f"特徴量抽出スクリプトが見つかりません: {extract_script}")
        return False
    
    # 特徴量抽出コマンドの構築
    cmd = [
        'python', extract_script,
        '--dataset', dataset_dir,
        '--f0_method', f0_method
    ]
    
    logger.info(f"実行コマンド: {' '.join(cmd)}")
    
    try:
        # 特徴量抽出プロセスの実行
        logger.info("特徴量抽出スクリプトを実行中...")
        result = subprocess.run(
            cmd, 
            check=True, 
            capture_output=False,
            text=True
        )
        logger.info("特徴量抽出が正常に完了しました")
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"特徴量抽出スクリプトの実行に失敗しました: {e}")
        return False
        
    except Exception as e:
        logger.error(f"予期しないエラーが発生しました: {e}")
        return False

def get_training_status(out_dir):
    """学習の進行状況を確認"""
    logger.info(f"学習状況を確認中: {out_dir}")
    
    if not os.path.exists(out_dir):
        logger.warning(f"出力ディレクトリが存在しません: {out_dir}")
        return None
    
    # 学習ログファイルの確認
    log_files = list(Path(out_dir).glob("*.log"))
    if not log_files:
        logger.info("学習ログファイルが見つかりません")
        return None
    
    # 最新のログファイルを確認
    latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
    logger.info(f"最新のログファイル: {latest_log}")
    
    try:
        with open(latest_log, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # 最後の数行を表示
            last_lines = lines[-10:] if len(lines) > 10 else lines
            logger.info("最新のログ内容:")
            for line in last_lines:
                logger.info(f"  {line.strip()}")
    except Exception as e:
        logger.error(f"ログファイルの読み込みに失敗: {e}")
    
    return latest_log

def cleanup_temp_files(temp_dir):
    """一時ファイルのクリーンアップ"""
    logger.info(f"一時ファイルのクリーンアップを開始: {temp_dir}")
    
    if not os.path.exists(temp_dir):
        logger.info("一時ディレクトリが存在しません")
        return
    
    try:
        temp_files = list(Path(temp_dir).glob("*"))
        for temp_file in temp_files:
            if temp_file.is_file():
                temp_file.unlink()
                logger.debug(f"削除: {temp_file}")
            elif temp_file.is_dir():
                import shutil
                shutil.rmtree(temp_file)
                logger.debug(f"削除: {temp_file}")
        
        logger.info("一時ファイルのクリーンアップが完了しました")
        
    except Exception as e:
        logger.error(f"一時ファイルのクリーンアップに失敗: {e}")
