from .. import DecodeModel

from pathlib import Path

import torch
import torch.nn.functional as F

from ..with_torch.weights import EMBED, FINAL_NORM, LM_HEAD, Qwen35Weights
from ..with_torch.config import Qwen35Config


class Qwen35Xyhs(DecodeModel):
    def __init__(
        self, model_dir: str | Path, device: str | torch.device = "cuda"
    ) -> None:
        self.cfg = Qwen35Config.from_model_dir(model_dir)
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.weights = Qwen35Weights(model_dir)

        self.embed = self.weights.get(EMBED)
        self.lm_head = self.weights.get(LM_HEAD)
        self.final_norm = self.weights.get(FINAL_NORM)

        self.num_layers = self.cfg.num_hidden_layers

    def forward(
        self, input_ids: torch.Tensor | list[int], position_offset: list[int] | None = None
    ):
        from .full_attn import Rope
        
        """只做 decode"""
        Rope.build_cos_sin(torch.as_tensor(position_offset, dtype=torch.long))
        ids = torch.as_tensor(input_ids, dtype=torch.long)
        if ids.dim() == 2:
            assert (
                ids.shape[1] == 1
            ), f"Only support decode, got ids length {ids.shape[1]}"
            ids = ids[None, :]
        assert (
            ids.dim() == 1
        ), f"Input ids must be 1-D or 2-D, got shape {tuple(ids.shape)}"
        x = self.embed[ids.cpu()].to(self.device).to(self.dtype)

        
