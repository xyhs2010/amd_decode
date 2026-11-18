"""Turn THUDM/LongBench-v2 data.json into prefill-ready JSONL.

    python prefill/build_longbench_v2.py

Outputs (next to data.json):

    metadata.jsonl     all 503 samples: labels, choices, token counts
    prompts_256k.jsonl samples whose prompt fits the 262144-token cap, with
                       the official template and a ``length`` bucket, ready
                       for ``prefill_qwen35.py --benchmark jsonl``

The prompt is the official LongBench v2 template; the answer prefix is
deliberately not included, matching the RULER cache protocol.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer

TEMPLATE = """Please read the following text and answer the question below.

<text>
{context}
</text>

What is the correct answer to this question:
{question}

Choices:
(A) {a}
(B) {b}
(C) {c}
(D) {d}

Format your response as follows: "The correct answer is (insert answer here)"."""

TASKS = {
    "Single-Document QA": "single_doc_qa",
    "Multi-Document QA": "multi_doc_qa",
    "Long In-context Learning": "long_icl",
    "Long-dialogue History Understanding": "long_dialogue",
    "Code Repository Understanding": "code_repo",
    "Long Structured Data Understanding": "long_structured",
}


def duration_bucket(tokens: int) -> int:
    if tokens <= 32768:
        return 32768
    if tokens <= 131072:
        return 131072
    return 262144


def build_prompt(row: dict) -> str:
    return TEMPLATE.format(
        context=row["context"],
        question=row["question"],
        a=row["choice_A"], b=row["choice_B"],
        c=row["choice_C"], d=row["choice_D"],
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", default="data/raw/longbench_v2/data.json")
    ap.add_argument("--tokenizer", default="data/models/Qwen3.5-9B/tokenizer.json")
    ap.add_argument("--max-tokens", type=int, default=262144)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    src = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else src.parent
    tok = Tokenizer.from_file(args.tokenizer)
    rows = json.loads(src.read_text())

    meta_path = out_dir / "metadata.jsonl"
    prompt_path = out_dir / f"prompts_{args.max_tokens // 1024}k.jsonl"
    fits = 0

    with meta_path.open("w", encoding="utf-8") as meta_f, \
            prompt_path.open("w", encoding="utf-8") as prompt_f:
        for index, row in enumerate(rows):
            sample_id = f"lb2_{index:03d}"
            prompt = build_prompt(row)
            # Contexts above 2M characters are far past 256K tokens even at a
            # pessimistic 8 chars/token; skip the expensive encode.
            if len(prompt) > 2_000_000:
                tokens = None
            else:
                tokens = len(tok.encode(prompt, add_special_tokens=False).ids)
            ok = tokens is not None and tokens <= args.max_tokens
            fits += ok

            meta = {
                "_id": row["_id"],
                "sample_id": sample_id,
                "domain": row["domain"],
                "task": TASKS.get(row["domain"], row["domain"]),
                "sub_domain": row["sub_domain"],
                "difficulty": row["difficulty"],
                "bucket": row["length"],
                "answer": row["answer"],
                "question": row["question"],
                "choices": {k: row[f"choice_{k}"] for k in "ABCD"},
                "tokens": tokens,
                "fits_256k": ok,
                "context_chars": len(row["context"]),
            }
            meta_f.write(json.dumps(meta, ensure_ascii=False) + "\n")

            if ok:
                prompt_f.write(json.dumps({
                    "sample_id": sample_id,
                    "task": TASKS.get(row["domain"], row["domain"]),
                    "length": duration_bucket(tokens),
                    "bucket": row["length"],
                    "tokens": tokens,
                    "prompt": prompt,
                }, ensure_ascii=False) + "\n")

            if (index + 1) % 50 == 0:
                print(f"{index + 1}/{len(rows)} fit={fits}")

    print(f"wrote {meta_path} and {prompt_path}: {fits}/{len(rows)} fit")


if __name__ == "__main__":
    main()
