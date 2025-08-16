from dataclasses import dataclass, asdict, field
import yaml
import os
from typing import Optional, List
from pathlib import Path

@dataclass
class AudioConfig:
    """音声処理設定"""
    sample_rate: int = 32000
    channels: int = 1
    normalize_lufs: float = -23.0
    trim_silence: bool = True
    vad_aggressiveness: int = 2
    chunk_duration: float = 12.0
    fade_in_ms: int = 100
    fade_out_ms: int = 100

@dataclass
class TrainingConfig:
    """学習設定"""
    batch_size: int = 4
    learning_rate: float = 0.0001
    steps: int = 20000
    save_every_n: int = 1000
    fp16: bool = True
    index_rate: float = 0.75
    f0_method: str = "rmvpe"
    gpu_id: int = 0
    num_workers: int = 4
    pin_memory: bool = True

@dataclass
class InferenceConfig:
    """推論設定"""
    transpose: int = 0
    f0_method: str = "rmvpe"
    rms_mix_rate: float = 0.25
    filter_radius: int = 3
    resample_sr: int = 0
    protect: float = 0.33
    index_rate: float = 0.75

@dataclass
class RVCConfig:
    """RVC設定のメインクラス"""
    # 基本設定
    project_name: str = "rvc_project"
    model_name: str = "rvc_model"
    
    # 音声設定
    audio: AudioConfig = field(default_factory=AudioConfig)
    
    # 学習設定
    training: TrainingConfig = field(default_factory=TrainingConfig)
    
    # 推論設定
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    
    # パス設定
    dataset_dir: str = "data/chunks"
    output_dir: str = "outputs"
    models_dir: str = "models"
    temp_dir: str = "temp"
    
    # 互換性のための古いパラメータ
    @property
    def sr(self) -> int:
        return self.audio.sample_rate
    
    @property
    def f0_method(self) -> str:
        return self.training.f0_method
    
    @property
    def batch(self) -> int:
        return self.training.batch_size
    
    @property
    def steps(self) -> int:
        return self.training.steps
    
    @property
    def fp16(self) -> bool:
        return self.training.fp16

    @staticmethod
    def load(path: str) -> "RVCConfig":
        """設定ファイルを読み込み"""
        if not os.path.exists(path):
            raise FileNotFoundError(f"設定ファイルが見つかりません: {path}")
        
        try:
            with open(path, "r", encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            # 古い形式の設定ファイルとの互換性
            if data and isinstance(data, dict):
                # 新しい形式に変換
                config = RVCConfig()
                
                # 基本設定
                if 'project_name' in data:
                    config.project_name = data['project_name']
                if 'model_name' in data:
                    config.model_name = data['model_name']
                
                # 音声設定
                if 'audio' in data:
                    audio_data = data['audio']
                    config.audio = AudioConfig(**audio_data)
                else:
                    # 古い形式からの変換
                    if 'sr' in data:
                        config.audio.sample_rate = data['sr']
                    if 'normalize_lufs' in data:
                        config.audio.normalize_lufs = data['normalize_lufs']
                
                # 学習設定
                if 'training' in data:
                    training_data = data['training']
                    config.training = TrainingConfig(**training_data)
                else:
                    # 古い形式からの変換
                    if 'batch' in data:
                        config.training.batch_size = data['batch']
                    if 'steps' in data:
                        config.training.steps = data['steps']
                    if 'fp16' in data:
                        config.training.fp16 = data['fp16']
                    if 'f0_method' in data:
                        config.training.f0_method = data['f0_method']
                
                # 推論設定
                if 'inference' in data:
                    inference_data = data['inference']
                    config.inference = InferenceConfig(**inference_data)
                else:
                    # 古い形式からの変換
                    if 'transpose' in data:
                        config.inference.transpose = data['transpose']
                    if 'rms_mix_rate' in data:
                        config.inference.rms_mix_rate = data['rms_mix_rate']
                
                # パス設定
                if 'dataset_dir' in data:
                    config.dataset_dir = data['dataset_dir']
                if 'output_dir' in data:
                    config.output_dir = data['output_dir']
                if 'models_dir' in data:
                    config.models_dir = data['models_dir']
                
                return config
            else:
                return RVCConfig()
                
        except yaml.YAMLError as e:
            raise ValueError(f"設定ファイルの解析に失敗しました: {e}")
        except Exception as e:
            raise ValueError(f"設定ファイルの読み込みに失敗しました: {e}")

    def save(self, path: str):
        """設定ファイルを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # 設定データを辞書に変換
        config_dict = asdict(self)
        
        # コメント付きで保存
        yaml_content = f"""# RVC設定ファイル
# プロジェクト名: {self.project_name}
# モデル名: {self.model_name}

{self._dict_to_yaml(config_dict)}"""
        
        with open(path, "w", encoding='utf-8') as f:
            f.write(yaml_content)

    def _dict_to_yaml(self, data: dict, indent: int = 0) -> str:
        """辞書をYAML形式の文字列に変換"""
        lines = []
        for key, value in data.items():
            if isinstance(value, dict):
                lines.append(f"{'  ' * indent}{key}:")
                lines.append(self._dict_to_yaml(value, indent + 1))
            elif isinstance(value, list):
                lines.append(f"{'  ' * indent}{key}:")
                for item in value:
                    if isinstance(item, dict):
                        lines.append(self._dict_to_yaml(item, indent + 1))
                    else:
                        lines.append(f"{'  ' * (indent + 1)}- {item}")
            else:
                lines.append(f"{'  ' * indent}{key}: {value}")
        return '\n'.join(lines)

    def validate(self) -> List[str]:
        """設定の妥当性を検証"""
        errors = []
        
        # 音声設定の検証
        if self.audio.sample_rate not in [16000, 32000, 44100, 48000]:
            errors.append(f"サンプリングレートは16kHz, 32kHz, 44.1kHz, 48kHzのいずれかである必要があります: {self.audio.sample_rate}")
        
        if self.audio.channels not in [1, 2]:
            errors.append(f"チャンネル数は1または2である必要があります: {self.audio.channels}")
        
        if not -70 <= self.audio.normalize_lufs <= -10:
            errors.append(f"LUFS正規化値は-70から-10の範囲である必要があります: {self.audio.normalize_lufs}")
        
        if not 0 <= self.audio.vad_aggressiveness <= 3:
            errors.append(f"VADアグレッシブネスは0から3の範囲である必要があります: {self.audio.vad_aggressiveness}")
        
        # 学習設定の検証
        if self.training.batch_size <= 0:
            errors.append(f"バッチサイズは正の値である必要があります: {self.training.batch_size}")
        
        if self.training.learning_rate <= 0:
            errors.append(f"学習率は正の値である必要があります: {self.training.learning_rate}")
        
        if self.training.steps <= 0:
            errors.append(f"学習ステップ数は正の値である必要があります: {self.training.steps}")
        
        if self.training.f0_method not in ["rmvpe", "crepe", "harvest", "pm"]:
            errors.append(f"F0抽出方法はrmvpe, crepe, harvest, pmのいずれかである必要があります: {self.training.f0_method}")
        
        # 推論設定の検証
        if not -12 <= self.inference.transpose <= 12:
            errors.append(f"音程シフトは-12から12の範囲である必要があります: {self.inference.transpose}")
        
        if not 0 <= self.inference.rms_mix_rate <= 1:
            errors.append(f"RMSミックス率は0から1の範囲である必要があります: {self.inference.rms_mix_rate}")
        
        if not 0 <= self.inference.protect <= 1:
            errors.append(f"プロテクト値は0から1の範囲である必要があります: {self.inference.protect}")
        
        return errors

    def get_absolute_paths(self, base_dir: str) -> "RVCConfig":
        """相対パスを絶対パスに変換"""
        config = RVCConfig()
        config.__dict__.update(self.__dict__)
        
        # パスを絶対パスに変換
        config.dataset_dir = os.path.abspath(os.path.join(base_dir, self.dataset_dir))
        config.output_dir = os.path.abspath(os.path.join(base_dir, self.output_dir))
        config.models_dir = os.path.abspath(os.path.join(base_dir, self.models_dir))
        config.temp_dir = os.path.abspath(os.path.join(base_dir, self.temp_dir))
        
        return config

    @classmethod
    def create_default(cls, project_name: str = "rvc_project") -> "RVCConfig":
        """デフォルト設定を作成"""
        config = cls()
        config.project_name = project_name
        config.model_name = f"{project_name}_model"
        return config

    def to_dict(self) -> dict:
        """設定を辞書形式で返す"""
        return asdict(self)
