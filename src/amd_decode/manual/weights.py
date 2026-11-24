from ..with_torch.weights import (
    EMBED,
    FINAL_NORM,
    LM_HEAD,
    _COMMON,
    _FULL_ATTENTION,
    _LINEAR_ATTENTION,
)
from pathlib import Path
import torch



class Qwen35WeightsXyhs:
    def __init__(self, model_dir: str | Path) -> None:
        self.state_dict = self.load(model_dir)

    def load(self, model_dir: str | Path) -> dict[str, torch.Tensor]:
        import json
        from safetensors.torch import load_file

        model_dir = Path(model_dir)
        index = json.loads((model_dir / "model.safetensors.index.json").read_text())
        shard_files: set[str] = set(index["weight_map"].values())
        state_dict = {}
        for shard_file in shard_files:
            shard_path = model_dir / shard_file
            state_dict.update(load_file(shard_path, device='cpu'))
        return state_dict
