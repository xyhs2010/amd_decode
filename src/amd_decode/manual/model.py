from .. import DecodeModel

from pathlib import Path

import torch
import torch.nn.functional as F

from ..with_torch.weights import EMBED, FINAL_NORM, LM_HEAD, Qwen35Weights
from .ffn import FFN
from .full_attn import FullAttn, Rope, KVCache
from .linear_attn import DeltaNet
from .function import rms
from safetensors.torch import load_file

NUM_LAYERS = 32
LAYER_PREFIX = "model.language_model.layers"


class Qwen35Xyhs(DecodeModel):
    def __init__(
        self,
        model_dir: str | Path,
        kv_dir: str | Path,
        device: str | torch.device = "cuda",
        max_len: int = 262144,
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.weights = Qwen35Weights(model_dir)

        self.ffn_list = [
            FFN(f"{LAYER_PREFIX}.{lid}", self.weights) for lid in range(NUM_LAYERS)
        ]
        self.attn_list: list[FullAttn | DeltaNet] = []
        kv_dir = Path(kv_dir)
        for lid in range(NUM_LAYERS):
            prefix = f"{LAYER_PREFIX}.{lid}"
            kv_dict = load_file(kv_dir / f"layer_{lid:02d}.safetensors")
            if lid % 4 == 3:
                kv_cache = KVCache(kv_dict, max_len)
                attn = FullAttn(prefix, self.weights, kv_cache)
            else:
                c_state, r_state = kv_dict["conv_state"], kv_dict["recurrent_state"]
                attn = DeltaNet(prefix, self.weights, c_state, r_state)
            self.attn_list.append(attn)

        self.embed = self.weights.get(EMBED)
        self.lm_head = self.weights.get(LM_HEAD)
        self.final_norm = self.weights.get(FINAL_NORM)

    def forward(
        self,
        input_ids: torch.Tensor | list[int],
        position_offset: list[int] | None = None,
    ):
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

        for lid in range(NUM_LAYERS):
            ffn, attn = self.ffn_list[lid], self.attn_list[lid]
            input = rms(x, ffn.wmap["input_layernorm.weight"])
            out = attn.forward(input)
            x = x + out
            x = rms(x, ffn.wmap["post_attention_layernorm.weight"])

            out = ffn.forward(x)
            x = x + out
