"""Runs a batch of chat prompts through a 14B model on a one-shot Colab T4.

`colab run` releases the VM when the script ends, so nothing can be fetched afterwards. Input therefore
travels inside this script (scripts/silver_colab.py fills PAYLOAD) and results come back on stdout as
gzip+base64 lines, flushed in batches so a dropped connection loses only the last batch.
"""
import base64
import concurrent.futures as futures
import gzip
import json
import os
import subprocess
import sys
import time
import urllib.request

PAYLOAD = "__PAYLOAD__"
DEADLINE_S = float("__DEADLINE_S__")
MODEL = "qwen2.5:14b-instruct-q4_K_M"
PARALLEL = 4
FLUSH_EVERY = 20
BASE = "http://127.0.0.1:11434"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def require_t4():
    gpus = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout
    log(gpus.strip())
    if "T4" not in gpus:
        sys.exit("ABORT: not a T4")  # an unrecognised --gpu falls back to an A100


def start_ollama():
    subprocess.run("apt-get install -y -qq zstd >/dev/null 2>&1; curl -fsSL https://ollama.com/install.sh | sh >/dev/null 2>&1", shell=True, check=True)
    env = {**os.environ, "OLLAMA_NUM_PARALLEL": str(PARALLEL), "OLLAMA_KEEP_ALIVE": "-1", "OLLAMA_FLASH_ATTENTION": "1"}
    subprocess.Popen(["ollama", "serve"], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(60):
        try:
            urllib.request.urlopen(BASE, timeout=2)
            break
        except OSError:
            time.sleep(1)
    subprocess.run(["ollama", "pull", MODEL], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    log("model ready")


def chat(task):
    body = {
        "model": MODEL,
        "stream": False,
        "messages": task["messages"],
        "options": {"temperature": task.get("temperature", 0.2), "num_predict": task.get("max_tokens", 400), "num_ctx": 4096},
    }
    if task.get("json"):
        body["format"] = "json"
    request = urllib.request.Request(f"{BASE}/api/chat", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.load(response)
    return {"id": task["id"], "text": data["message"]["content"], "tokens": data.get("eval_count", 0)}


def emit(rows):
    packed = base64.b64encode(gzip.compress(json.dumps(rows, ensure_ascii=False).encode())).decode()
    print(f"@@RESULT {packed}", flush=True)


def main():
    started = time.time()
    require_t4()
    tasks = json.loads(gzip.decompress(base64.b64decode(PAYLOAD)))
    log(f"{len(tasks)} tasks")
    start_ollama()
    log(f"setup took {time.time() - started:.0f}s")

    batch, done, tokens, failed = [], 0, 0, 0
    run_started = time.time()
    with futures.ThreadPoolExecutor(PARALLEL) as pool:
        pending = {pool.submit(chat, t) for t in tasks[:PARALLEL * 2]}
        queue = iter(tasks[PARALLEL * 2:])
        while pending:
            finished, pending = futures.wait(pending, return_when=futures.FIRST_COMPLETED)
            for item in finished:
                try:
                    row = item.result()
                    tokens += row["tokens"]
                    batch.append(row)
                    done += 1
                except Exception as error:
                    failed += 1
                    log(f"task failed: {error}")
                if time.time() - started < DEADLINE_S:
                    following = next(queue, None)
                    if following is not None:
                        pending.add(pool.submit(chat, following))
            if len(batch) >= FLUSH_EVERY:
                emit(batch)
                batch = []
                elapsed = time.time() - run_started
                log(f"{done}/{len(tasks)} done, {tokens / elapsed:.0f} tok/s, {elapsed:.0f}s")
    if batch:
        emit(batch)
    log(f"finished {done}/{len(tasks)}, failed {failed}, total {time.time() - started:.0f}s")
    print("@@END", flush=True)


main()
