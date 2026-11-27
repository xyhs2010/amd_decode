from .. import DecodeModel

from pathlib import Path

import torch

from .weights import EMBED, LM_HEAD, FINAL_NORM, Qwen35Weights
from .ffn import FFN
from .full_attn import FullAttn, Rope
from .linear_attn import DeltaNet
from .function import rms
from safetensors.torch import load_file

NUM_LAYERS = 32
LAYER_PREFIX = "model.language_model.layers"


class Qwen35Xyhs(DecodeModel):
    def __init__(
        self,
        model_dir: str | Path,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.dtype = torch.float16
        self.weights = Qwen35Weights(model_dir)

        self.ffn_list = [
            FFN(f"{LAYER_PREFIX}.{lid}", self.weights) for lid in range(NUM_LAYERS)
        ]
        self.attn_list = [
            (FullAttn if lid % 4 == 3 else DeltaNet)(
                f"{LAYER_PREFIX}.{lid}", self.weights
            )
            for lid in range(NUM_LAYERS)
        ]

        self.embed = self.weights.state_dict[EMBED]
        self.lm_head = self.weights.state_dict[LM_HEAD]
        self.final_norm = self.weights.state_dict[FINAL_NORM]

    def load_prefill(self, kv_dir: str = None, max_len: int | None = None) -> None:
        kv_dir = Path(kv_dir)
        for lid in range(NUM_LAYERS):
            kv_dict = load_file(kv_dir / f"layer_{lid:02d}.safetensors")
            self.attn_list[lid].load_prefill(kv_dict, max_len)

    def forward(
        self,
        input_ids: torch.Tensor | list[int],
        position_offset: list[int] | None = None,
    ):
        """只做 decode"""
        if position_offset == None:
            for attn in self.attn_list:
                if isinstance(attn, FullAttn):
                    position_offset = [attn.kv_cache.offset] * len(input_ids)
                    break
        Rope.build_cos_sin(torch.as_tensor(position_offset, dtype=torch.long).cuda())
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
            # if lid == 0:
            #     print("attn", out)
            x = x + out

            out = ffn.forward(rms(x, ffn.wmap["post_attention_layernorm.weight"]))
            x = x + out

            # if lid == 0:
            #     print("mlp", x)
