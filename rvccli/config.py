from dataclasses import dataclass, asdict
import yaml

@dataclass
class RVCConfig:
    sr: int = 32000
    f0_method: str = "rmvpe"
    batch: int = 4
    steps: int = 20000
    fp16: bool = True
    # 他のパラメータも必要に応じて追加

    @staticmethod
    def load(path: str) -> "RVCConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return RVCConfig(**data)

    def save(self, path: str):
        with open(path, "w") as f:
            yaml.safe_dump(asdict(self), f, allow_unicode=True)
