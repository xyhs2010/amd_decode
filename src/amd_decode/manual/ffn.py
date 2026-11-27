import torch
from .weights import Qwen35Weights, _COMMON
from .int4 import Q4Weight
from .function import rms, linear_int4


class FFN(object):
    def __init__(self, prefix: str, wprovider: Qwen35Weights):
        self.prefix = prefix
        self.wmap: dict[str, Q4Weight] = {}
        for wname in _COMMON:
            weights = wprovider.state_dict[f"{prefix}.{wname}"]
            if wname.startswith("mlp"):
                q4_weights = Q4Weight(weights)
                q4_weights.to_gpu()
                self.wmap[wname] = q4_weights
            else:
                self.wmap[wname] = weights.cuda()

    def forward(self, x: torch.Tensor):
        import torch.nn.functional as F

        gate = F.silu(linear_int4(x, self.wmap["mlp.gate_proj.weight"]))
        up = linear_int4(x, self.wmap["mlp.up_proj.weight"])
        y = linear_int4(gate * up, self.wmap["mlp.down_proj.weight"])
        return y
