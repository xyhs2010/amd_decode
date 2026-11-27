import torch
import torch.nn.functional as F
from .weights import Qwen35Weights, _LINEAR_ATTENTION

NUM_LINEAR_KHEAD = 16
NUM_LINEAR_VHEAD = 32
LINEAR_DIM = 128
LINEAR_KERNEL_DIM = 4


class DeltaNet(object):
    def __init__(
        self,
        prefix: str,
        wprovider: Qwen35Weights,
    ):
        self.prefix = prefix
        self.wmap: dict[str, torch.Tensor] = {}
        for wname in _LINEAR_ATTENTION:
            weights = wprovider.state_dict[f"{prefix}.{wname}"]
            self.wmap[wname] = weights.to(dtype=torch.float16, device="cuda")
        self.conv_buf = self.recurrent_state = None

    def load_prefill(
        self, kv_dict: dict[str, torch.Tensor], max_len: int | None = None
    ):
        conv_state = kv_dict["conv_state"].to(dtype=torch.float16, device="cuda")
        shape = list(conv_state.shape)
        shape[-1] = 1
        self.conv_buf = torch.cat([conv_state, conv_state.new_zeros(shape)], dim=-1)
        self.recurrent_state = kv_dict["recurrent_state"].half().cuda()

    def forward(self, x: torch.Tensor):
        from .function import rms_gated

        batch = x.shape[0]
        conved = self.conv(x)
        out = self.delta_rule(x, conved)

        z = F.linear(x, self.wmap["linear_attn.in_proj_z.weight"])
        z = z.view(batch, NUM_LINEAR_VHEAD, LINEAR_DIM)
        out = rms_gated(out.to(x.dtype), self.wmap["linear_attn.norm.weight"], z)
        out = F.linear(out.view(batch, -1), self.wmap["linear_attn.out_proj.weight"])
        return out

    def delta_rule(self, x: torch.Tensor, conved: torch.Tensor):
        from math import sqrt

        batch = x.shape[0]
        conved = conved.view(batch, -1, LINEAR_DIM)
        head_split_shape = (NUM_LINEAR_KHEAD, NUM_LINEAR_KHEAD, NUM_LINEAR_VHEAD)
        q, k, v = conved.float().split(head_split_shape, dim=-2)
        q = q * torch.rsqrt(q.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        q = q / sqrt(LINEAR_DIM)
        k = k * torch.rsqrt(k.pow(2).sum(dim=-1, keepdim=True) + 1e-6)
        k, q = k[:, :, None, None, :], q[:, :, None, None, :]

        if self.recurrent_state is None:
            self.recurrent_state = torch.zeros(
                (batch, NUM_LINEAR_VHEAD, LINEAR_DIM, LINEAR_DIM),
                dtype=torch.float32,
                device=q.device,
            )

        beta = F.linear(x, self.wmap["linear_attn.in_proj_b.weight"]).sigmoid().float()
        a = F.linear(x, self.wmap["linear_attn.in_proj_a.weight"])
        g = -self.wmap["linear_attn.A_log"].float().exp()
        g = g * F.softplus((a + self.wmap["linear_attn.dt_bias"]).float())

        state = self.recurrent_state
        state = state * g.exp()[..., None, None]
        state = state.view(batch, NUM_LINEAR_KHEAD, -1, LINEAR_DIM, LINEAR_DIM)
        kv_mem = (k @ state).view(batch, NUM_LINEAR_VHEAD, LINEAR_DIM)
        delta = (v - kv_mem) * beta[..., None]
        delta = delta.view(batch, NUM_LINEAR_KHEAD, -1, 1, LINEAR_DIM)
        state = state + k[:, :, :, 0, :, None] * delta
        out = (q @ state).view(batch, NUM_LINEAR_VHEAD, LINEAR_DIM)
        self.recurrent_state = state.view(batch, -1, LINEAR_DIM, LINEAR_DIM)
        return out

    def conv(self, x: torch.Tensor):
        batch = x.shape[0]
        if self.conv_buf is None:
            mixed_dim = (NUM_LINEAR_KHEAD * 2 + NUM_LINEAR_VHEAD) * LINEAR_DIM
            self.conv_buf = torch.zeros(
                (batch, mixed_dim, LINEAR_KERNEL_DIM + 1),
                dtype=x.dtype,
                device=x.device,
            )

        mixed = F.linear(x, self.wmap["linear_attn.in_proj_qkv.weight"])
        self.conv_buf[:, :, -1] = mixed
        conv_w = self.wmap["linear_attn.conv1d.weight"].squeeze(1)
        conved = F.silu((self.conv_buf[:, :, 1:] * conv_w).sum(dim=-1))
        self.conv_buf[:, :, :-1] = self.conv_buf[:, :, 1:]

        return conved
