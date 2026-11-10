"""Trigger: re-run the pipeline when the data, code, or parameters change.

A fingerprint (MD5 of the raw data, params.yaml, and every Python file in src/,
api/, and pipeline/) is stored after each successful run. If the fingerprint is
unchanged, nothing runs.

    python -m pipeline.trigger                 # run once if something changed
    python -m pipeline.trigger --force         # run regardless
    python -m pipeline.trigger --watch 30      # poll every 30 seconds
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time

from pipeline.run_pipeline import git_info, main as run_pipeline
from src.config import ROOT, load_params
from src.data import file_md5

STATE_PATH = ROOT / ".pipeline_state.json"


def fingerprint() -> dict:
    code = hashlib.md5()
    files = sorted(p for d in ("src", "api", "pipeline") for p in (ROOT / d).rglob("*.py"))
    for p in files:
        code.update(p.read_bytes())
    return {
        "data": file_md5(ROOT / load_params()["data"]["raw_path"]),
        "params": file_md5(ROOT / "params.yaml"),
        "code": code.hexdigest(),
    }


def changed_parts(current: dict) -> list:
    if not STATE_PATH.exists():
        return ["first run"]
    last = json.loads(STATE_PATH.read_text()).get("fingerprint", {})
    return [k for k in current if current[k] != last.get(k)]


def check_and_run(force: bool = False) -> int:
    current = fingerprint()
    changes = ["forced"] if force else changed_parts(current)
    if not changes:
        print("No change in data, code, or params. Pipeline not triggered.")
        return 0
    print(f"Change detected in: {', '.join(changes)}. Triggering pipeline.")
    code = run_pipeline()
    if code == 0:  # only a successful run updates the state, so failures retry
        STATE_PATH.write_text(json.dumps({"fingerprint": current, "git": git_info(),
                                          "last_success": time.strftime("%Y-%m-%dT%H:%M:%S%z")}, indent=2))
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--watch", type=int, metavar="SECONDS", help="poll for changes at this interval")
    args = parser.parse_args()
    if not args.watch:
        return check_and_run(args.force)
    print(f"Watching for changes every {args.watch}s. Ctrl+C to stop.")
    while True:
        check_and_run()
        time.sleep(args.watch)


if __name__ == "__main__":
    sys.exit(main())
