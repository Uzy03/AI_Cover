import typer

app = typer.Typer(help="Retrieval-based Voice Conversion CLI")

@app.command()
def setup():
    """セットアップ: 依存導入・外部リポジトリclone・環境チェック"""
    pass

@app.command("download-models")
def download_models():
    """事前学習モデルのダウンロード"""
    import os
    from . import download_models as dm
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    models_dir = os.path.abspath(models_dir)
    dm.ensure_models(models_dir)

@app.command()
def prep(in_dir: str = typer.Option(..., help="入力ディレクトリ"), out_dir: str = typer.Option(..., help="出力ディレクトリ")):
    """音声前処理（32kHz/mono, 無音トリム, LUFS, 分割）"""
    pass

@app.command()
def train():
    """学習プロセスの起動"""
    pass

@app.command()
def infer(wav: str = typer.Option(..., help="入力wav"), out: str = typer.Option(..., help="出力wav")):
    """推論（音声変換）"""
    pass

@app.command()
def pack():
    """モデル一式のパッケージング"""
    pass

if __name__ == "__main__":
    app()
