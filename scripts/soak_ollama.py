"""Checks that the configured GPU split holds up across long, varied prompts.

Exits non-zero if decode speed drops below --min-decode, or the prompts in the last third run
slower than the first third, which is how a slow spill into shared memory would show up.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
import json
import statistics
import subprocess
from typing import Any, Dict, List

import httpx

from apps.config import ollama_request_fields, settings
from apps.services.llm_service import get_llm_model_name


def gpu_memory_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True,
    ).stdout
    return int(out.strip().splitlines()[0])


def build_prompts(contexts_path: str, count: int, target_chars: int) -> List[str]:
    with open(contexts_path, "r", encoding="utf-8") as f:
        records = [r for r in json.load(f)["records"] if r["top_chunks"]]
    texts = [c.get("raw_content") or "" for r in records for c in r["top_chunks"]]
    if not texts:
        raise SystemExit(f"No contexts found in {contexts_path}")

    prompts: List[str] = []
    for i in range(count):
        body, offset = "", i * 3
        while len(body) < target_chars:
            body += texts[offset % len(texts)] + "\n\n"
            offset += 1
        query = records[i % len(records)]["query"]
        prompts.append(f"NGỮ CẢNH PHÁP LÝ:\n{body[:target_chars]}\n\nCÂU HỎI:\n{query}\n\nTrả lời ngắn gọn.")
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contexts", default="results/retrieval/retrieval.json")
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--predict", type=int, default=150)
    parser.add_argument("--target-chars", type=int, default=7000)
    parser.add_argument("--min-decode", type=float, default=9.0)
    parser.add_argument("--out", default="results/soak/soak.json")
    args = parser.parse_args()

    host = os.getenv("OLLAMA_HOST", "127.0.0.1:11434")
    host = host if host.startswith("http") else f"http://{host}"
    rows: List[Dict[str, Any]] = []
    with httpx.Client(timeout=300.0) as client:
        for idx, prompt in enumerate(build_prompts(args.contexts, args.count, args.target_chars), 1):
            payload = {
                "model": get_llm_model_name(),
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                **ollama_request_fields(temperature=0.1, num_predict=args.predict),
            }
            r = client.post(f"{host}/api/chat", json=payload)
            r.raise_for_status()
            d = r.json()
            row = {
                "prompt_tokens": d.get("prompt_eval_count", 0),
                "prefill_tps": d.get("prompt_eval_count", 0) / max(d.get("prompt_eval_duration", 1) / 1e9, 1e-6),
                "decode_tokens": d.get("eval_count", 0),
                "decode_tps": d.get("eval_count", 0) / max(d.get("eval_duration", 1) / 1e9, 1e-6),
                "load_s": d.get("load_duration", 0) / 1e9,
                "vram_mib": gpu_memory_mib(),
            }
            rows.append(row)
            print(
                f"[{idx}/{args.count}] in {row['prompt_tokens']} prefill {row['prefill_tps']:.0f} t/s "
                f"decode {row['decode_tps']:.2f} t/s vram {row['vram_mib']} MiB",
                flush=True,
            )

    measured = rows[1:]  # the first request includes model load
    third = max(len(measured) // 3, 1)
    decode = [r["decode_tps"] for r in measured]
    first_third = statistics.median(decode[:third])
    last_third = statistics.median(decode[-third:])
    summary = {
        "num_gpu": settings.OLLAMA_NUM_GPU,
        "num_ctx": settings.OLLAMA_NUM_CTX,
        "decode_median": statistics.median(decode),
        "decode_min": min(decode),
        "decode_first_third": first_third,
        "decode_last_third": last_third,
        "prefill_median": statistics.median(r["prefill_tps"] for r in measured),
        "prefill_min": min(r["prefill_tps"] for r in measured),
        "vram_max_mib": max(r["vram_mib"] for r in measured),
        "reloads": sum(1 for r in measured if r["load_s"] > 2.0),
    }
    summary["passed"] = (
        summary["decode_median"] >= args.min_decode
        and last_third >= 0.85 * first_third
        and summary["reloads"] == 0
    )
    print(json.dumps(summary, indent=2), flush=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "rows": rows}, f, indent=2)
    sys.exit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
