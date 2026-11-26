import torch
from .weights import Qwen35Weights, _FULL_ATTENTION
from .function import rms

HEAD_DIM = 256
NUM_Q_HEAD = 16
NUM_KV_HEAD = 4
ROPE_DIM = 64
ROPE_THETA = 10000000


class Rope(object):
    cos: torch.Tensor = torch.empty(0)
    sin: torch.Tensor = torch.empty(0)

    @classmethod
    def build_cos_sin(
        cls,
        positions: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """``positions``: 1-D absolute positions. Returns cos/sin of shape [T, 64]."""
        inv_freq = 1.0 / (
            ROPE_THETA ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float32) / ROPE_DIM)
        )
        freqs = (
            positions.to(torch.float32)[:, None]
            * inv_freq.to(positions.device)[None, :]
        )
        emb = torch.cat((freqs, freqs), dim=-1)
        cls.cos, cls.sin = emb.cos().to(dtype), emb.sin().to(dtype)

    @classmethod
    def apply_rope(
        cls,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Rotate the first ``rope_dim`` channels of ``x`` [B, H, T, D]."""
        half = ROPE_DIM // 2
        x_rot = x[..., :ROPE_DIM]
        rotated = torch.cat((-x_rot[..., half:], x_rot[..., :half]), dim=-1)
        out = x.clone()
        out[..., :ROPE_DIM] = (
            x_rot * cls.cos[None, None] + rotated * cls.sin[None, None]
        )
        return out


class KVCache(object):
    def __init__(self, kv_dict: dict[str, torch.Tensor], max_len: int):
        k, v = kv_dict["k"], kv_dict["v"]
        shape = list(k.shape)
        self.offset = shape[2]
        shape[2] = max_len

        self._k = torch.empty(shape, dtype=k.dtype, device="cuda")
        self._v = torch.empty_like(self.k)
        self._k[:, :, : self.offset, :].copy_(k)
        self._v[:, :, : self.offset, :].copy_(v)

    def append(self, k: torch.Tensor, v: torch.Tensor):
        self._k[:, :, self.offset, :].copy_(k)
        self._v[:, :, self.offset, :].copy_(v)
        self.offset += 1

    @property
    def k(self) -> torch.Tensor:
        return self._k[:, :, : self.offset, :]

    @property
    def v(self) -> torch.Tensor:
        return self._v[:, :, : self.offset, :]


class FullAttn(object):
    def __init__(self, prefix: str, wprovider: Qwen35Weights, kv_cache: KVCache):
        self.prefix = prefix
        self.wmap: dict[str, torch.Tensor] = {}
        for wname in _FULL_ATTENTION:
            weights = wprovider.state_dict[f"{prefix}.{wname}"]
            self.wmap[wname] = weights.cuda()
        self.kv_cache = kv_cache

        self.qgkv_weight = torch.concat(
            [
                self.wmap["self_attn.q_proj.weight"],
                self.wmap["self_attn.k_proj.weight"],
                self.wmap["self_attn.v_proj.weight"],
            ],
            dim=0,
        )

    def forward(self, x: torch.Tensor):
        import torch.nn.functional as F
        from math import sqrt

        batch = x.shape[0]
        qgkv = F.linear(x, self.qgkv_weight)
        qgkv = qgkv.view(batch, 2 * (NUM_Q_HEAD + NUM_KV_HEAD), HEAD_DIM)
        qg, k, v = torch.split(qgkv, [2 * NUM_Q_HEAD, NUM_KV_HEAD, NUM_KV_HEAD], dim=1)
        q, gate = qg.view(batch, NUM_Q_HEAD, 2 * HEAD_DIM).chunk(2, dim=-1)

        q = rms(q, self.wmap["self_attn.q_norm.weight"])
        k = rms(k, self.wmap["self_attn.k_norm.weight"])

        q = Rope.apply_rope(q)
        k = Rope.apply_rope(k)

        self.kv_cache.append(k, v)

        ngroup = NUM_Q_HEAD // NUM_KV_HEAD
        qgroup = q.view(batch, NUM_KV_HEAD, ngroup, HEAD_DIM)
        scores = qgroup @ self.kv_cache.k.transpose(-1, -2) / sqrt(HEAD_DIM)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
        out = (probs @ self.kv_cache.v).view(batch, NUM_Q_HEAD * HEAD_DIM)
        out = out * torch.sigmoid(gate.view(batch, NUM_Q_HEAD * HEAD_DIM))
        out = F.linear(out, self.wmap["self_attn.o_proj.weight"])
        return out
