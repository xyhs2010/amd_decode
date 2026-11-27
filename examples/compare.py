from amd_decode.with_torch import Qwen35Torch
from amd_decode.utils.prefill_cache import SavedKVCache
from amd_decode.manual.model import Qwen35Xyhs
import torch


def main():
    model_dir = "data/models/Qwen3.5-9B"
    kv_dir = "data/kv_cache/Qwen3.5-9B-chat/bfloat16/jsonl/L32768/long_dialogue/lb2_086"

    # kv_cache = SavedKVCache(kv_dir)
    # torch_model = Qwen35Torch(model_dir)
    # torch_model.load_prefill(kv_cache, 33000)
    # torch_model.forward([4000])

    # del kv_cache, torch_model
    # torch.cuda.empty_cache()

    model = Qwen35Xyhs(model_dir)
    model.load_prefill(kv_dir, 33000)
    model.forward([4000])

    import ipdb
    ipdb.set_trace()


if __name__ == "__main__":
    main()
