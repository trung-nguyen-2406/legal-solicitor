"""Launcher for scripts/colab_llm_job.py: packs tasks, runs them on a one-shot T4, unpacks the results.

Always `--gpu T4`, never `--keep`, and no credentials are sent. The VM is released when the job ends.
"""
import argparse
import base64
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time

COLAB = os.path.expanduser("~/.local/bin/colab")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "colab_llm_job.py")


def balance() -> str:
    out = subprocess.run([COLAB, "usage"], capture_output=True, text=True, encoding="utf-8", errors="replace").stdout
    return " ".join(out.split())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, help="JSON list of {id, messages, ...}")
    parser.add_argument("--out", required=True, help="JSONL of {id, text, tokens}")
    parser.add_argument("--minutes", type=float, default=60, help="The job stops taking tasks after this; the VM ends 10 minutes later at most")
    parser.add_argument("--session", required=True)
    args = parser.parse_args()
    if not args.session.startswith("v51-"):
        sys.exit("session name must start with v51-")

    with open(args.tasks, "r", encoding="utf-8") as f:
        tasks = json.load(f)
    payload = base64.b64encode(gzip.compress(json.dumps(tasks, ensure_ascii=False).encode())).decode()
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        script = f.read().replace("__PAYLOAD__", payload).replace("__DEADLINE_S__", str(args.minutes * 60))

    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, "job.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(script)
        print("before:", balance(), flush=True)
        started = time.time()
        command = [COLAB, "run", "--gpu", "T4", "--session", args.session, "--timeout", str(args.minutes * 60 + 600), path]
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
        rows, ended = 0, False
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as out:
            for line in process.stdout:
                if "@@RESULT " in line:
                    packed = line.split("@@RESULT ", 1)[1].strip()
                    for row in json.loads(gzip.decompress(base64.b64decode(packed))):
                        out.write(json.dumps(row, ensure_ascii=False) + "\n")
                        rows += 1
                    out.flush()
                elif "@@END" in line:
                    ended = True
                else:
                    print(line.rstrip()[:300], flush=True)
        code = process.wait()
    print(f"exit {code}, rows {rows}/{len(tasks)}, clean end {ended}, {(time.time() - started) / 60:.1f} min", flush=True)
    print("after:", balance(), flush=True)


if __name__ == "__main__":
    main()
