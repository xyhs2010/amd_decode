# Qwen3.5-9B prefill

A thin wrapper around the stock `transformers` implementation: one forward with
`DynamicCache`, then the cache tensors are written to disk.  There is no custom
attention, no quantization and no verification — the saved cache is exactly
what transformers computed, and it is checked on the decode side.

## Environment

```bash
uv venv --python 3.12
uv pip install -r prefill/requirements.txt
```

`requirements.txt` pins `transformers>=5.23` (the first release with
`qwen3_5`).  `--device` accepts any torch device string and `--dtype` is
`bfloat16` or `float16`, so the script makes no assumption about the machine.

## Data

Model (19 GB):

```bash
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1
unset ALL_PROXY all_proxy HTTPS_PROXY HTTP_PROXY https_proxy http_proxy
python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('Qwen/Qwen3.5-9B', local_dir='data/models/Qwen3.5-9B')"
```

RULER data is the same tree the existing caches use:
`data/raw/ruler/L<length>/<task>/test.jsonl`.

## Run

```bash
# smoke test: 2 samples per (length, task)
python prefill/prefill_qwen35.py \
    --model data/models/Qwen3.5-9B \
    --benchmark ruler --lengths 16384 \
    --tasks niah_single_2 cwe --limit 2

# the full sweep
python prefill/prefill_qwen35.py \
    --benchmark ruler --lengths 16384 32768 65536 131072 262144
```

`--chunk-size N` splits the forward into N-token pieces when the whole prompt
does not fit in one pass; the default `0` runs it in one forward.  Chunking
does not change the saved result.

`--limit` is per `(length, task)` for RULER and overall for `jsonl`.

Resumable: a sample is skipped when `meta.json` and all 32 `layer_*.safetensors`
exist.  Use `--overwrite` to rebuild.

The answer prefix is **not** part of the prefill; the decode side appends it,
exactly like the RULER protocol.

## Output layout

```
<out>/<model_tag>/<dtype>/<benchmark>/L<length>/<task>/<sample_id>/
    meta.json
    tokens.safetensors
    layer_00.safetensors ... layer_31.safetensors
```

`model_tag` defaults to the model directory name.  Per layer:

- linear attention (24 layers): `conv_state` `[1, 8192, 4]` and
  `recurrent_state` `[1, 32, 128, 128]` (fp32)
- full attention (8 layers): `k` and `v` `[1, 4, T, 256]` in the model dtype

`meta.json` records `seq_len`, `layer_types`, the head dimensions, the rope
parameters and the linear-attention dimensions, so the decode reader does not
depend on these constants.

## LongBench v2

```bash
mkdir -p data/raw/longbench_v2
curl -L -C - -o data/raw/longbench_v2/data.json \
  https://hf-mirror.com/datasets/THUDM/LongBench-v2/resolve/main/data.json

python prefill/build_longbench_v2.py
```

This writes `metadata.jsonl` (all 503 samples, with answers and choices) and
`prompts_256k.jsonl` (the official template, only samples that fit 262144
tokens): 402 of 503 samples.

Then prefill the fitting subset:

```bash
python prefill/prefill_qwen35.py \
    --benchmark jsonl --jsonl data/raw/longbench_v2/prompts_256k.jsonl
```

The chat-template question (raw official prompt vs the model's chat template)
is decided with the benchmark adapter, not here; `build_longbench_v2.py` emits
the raw official template.

## Other benchmarks

Any JSONL works with `--benchmark jsonl`; one object per line:

```json
{"sample_id": "...", "task": "single_doc_qa", "length": 0, "prompt": "full prompt text"}
```

`prompt` must be the **entire** text to prefill.  `length` is only used for the
directory name.

## Limits

- Prompt tokens above `--max-tokens 262144` are skipped, matching the model's
  native context.
- The cache is stored in the model dtype; there is no q8/q4 step here.
