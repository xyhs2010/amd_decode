# Qwen3.5-9B prefill

Builds the dense prefill cache used by the decode backend.  This is the one
place where `transformers` is the right tool: it already implements Gated
DeltaNet, mRoPE and the hybrid attention; the decode path stays dependency-free.

Run it on a CUDA machine.  A 32 GB card (e.g. 5090) is enough for the bf16
model with q8 K/V and `--chunk-size 4096`.

## Environment

```bash
uv venv --python 3.12
uv pip install -r prefill/requirements.txt
```

`requirements.txt` pins `transformers>=5.23` (the first release with
`qwen3_5`).

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
# small smoke test: 2 samples per (length, task)
python prefill/prefill_qwen35.py \
    --model data/models/Qwen3.5-9B \
    --benchmark ruler --lengths 16384 \
    --tasks niah_single_2 cwe --limit 2 \
    --out data/kv_cache --verify

# the real sweep: 16K..256K, q8 K/V
python prefill/prefill_qwen35.py \
    --benchmark ruler --lengths 16384 32768 65536 131072 262144 \
    --out data/kv_cache --chunk-size 4096
```

`--limit` is per `(length, task)` for RULER and overall for `jsonl`.

Resumable: a sample is skipped when `meta.json` and all 32 `layer_*.safetensors`
exist.  Use `--overwrite` to rebuild.

The answer prefix is **not** part of the prefill; the decode side appends it,
exactly like the existing RULER protocol.

## Output layout

```
<out>/<model_tag>/<quant>/<benchmark>/L<length>/<task>/<sample_id>/
    meta.json
    tokens.safetensors
    layer_00.safetensors ... layer_31.safetensors
```

`model_tag` defaults to `<model dir name>-bf16`.  Per layer:

- linear attention (24 layers): `conv_state` fp16 `[1, 8192, 4]` and
  `recurrent_state` fp32 `[1, 32, 128, 128]`
- full attention (8 layers, q8): `k_q`/`v_q` int8 `[1, 4, T, 256]`,
  `k_scale` float32 `[1, 4, ceil(T/8192), 1, 256]`, `v_scale` float32
  `[1, 4, T, 1]`

q8 at 256K is about 4.3 GB per sample; the DeltaNet state adds ~51 MB.

## Other benchmarks

Any JSONL works with `--benchmark jsonl`; one object per line:

```json
{"sample_id": "...", "task": "single_doc_qa", "length": 0, "prompt": "full prompt text"}
```

`prompt` must be the **entire** text to prefill (for LongBench v2: context,
question, choices and the answer-format instruction).  `length` is only used
for the directory name.

## Memory notes

- The model is bf16 (`~19 GB`); the script does not quantize weights.
- Prompt tokens above `--max-tokens 262144` are skipped, matching the model's
  native context.
- If the first sample OOMs, lower `--chunk-size`; the attention mask grows with
  `past + chunk`, and DeltaNet's chunked kernel allocates per chunk.
- `--quant fp16` doubles K/V memory and disk; q8 is the intended default.
