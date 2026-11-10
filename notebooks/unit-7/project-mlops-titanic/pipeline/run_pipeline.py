"""CI/CD pipeline: validate -> test -> train -> gate -> deploy -> smoke test -> monitor.

Each step is a subprocess. The pipeline stops at the first failure, so a bad
data file, a failing test, or a model that misses the quality gate never
reaches the serving artifact (models/model.joblib). The API watches that file
and hot-reloads it, so a successful run is a deployment.

    python -m pipeline.run_pipeline

A JSON record of every run (git commit, data hash, step results) is written to
reports/pipeline_runs/.
"""
from __future__ import annotations

import json
import subprocess
import sys
import time
from datetime import datetime, timezone

from src.config import METRICS_PATH, PIPELINE_RUNS_DIR, ROOT, load_params
from src.data import file_md5

PY = sys.executable

STEPS = [
    ("validate_and_prepare_data", [PY, "-m", "src.data"]),
    ("unit_tests", [PY, "-m", "pytest", "-q", "tests/test_data.py", "tests/test_features.py"]),
    ("eda", [PY, "-m", "src.eda"]),
    ("train_gate_and_promote", [PY, "-m", "src.train"]),
    ("model_and_api_tests", [PY, "-m", "pytest", "-q", "tests/test_model.py", "tests/test_api.py"]),
    ("monitoring", [PY, "-m", "src.monitor"]),
]


def git_info() -> dict:
    """Commit, branch, and dirty flag via GitPython, so every run is traceable to code."""
    try:
        import git
        repo = git.Repo(ROOT, search_parent_directories=True)
        try:
            branch = repo.active_branch.name
        except TypeError:
            branch = "detached"
        return {"commit": repo.head.commit.hexsha, "branch": branch, "dirty": repo.is_dirty(untracked_files=False)}
    except Exception:
        return {"commit": None, "branch": None, "dirty": None}


def main() -> int:
    started = datetime.now(timezone.utc)
    record = {
        "started_at": started.isoformat(timespec="seconds"),
        "git": git_info(),
        "data_md5": file_md5(ROOT / load_params()["data"]["raw_path"]),
        "steps": [],
        "status": "running",
    }
    print(f"Pipeline start | commit={record['git']['commit']} | data_md5={record['data_md5']}")

    ok = True
    for name, cmd in STEPS:
        t0 = time.time()
        print(f"\n=== {name} ===", flush=True)
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
        tail = proc.stdout.strip().splitlines()[-6:]
        if proc.returncode != 0:  # show the error stream only when a step fails
            tail += [l for l in proc.stderr.strip().splitlines() if "warn" not in l.lower()][-8:]
        print("\n".join(tail))
        record["steps"].append({"name": name, "status": "passed" if proc.returncode == 0 else "failed",
                                "seconds": round(time.time() - t0, 1), "output_tail": tail})
        if proc.returncode != 0:
            ok = False
            print(f"\nStep '{name}' failed. Stopping. The serving model was not changed.")
            break

    record["status"] = "succeeded" if ok else "failed"
    record["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if ok and METRICS_PATH.exists():
        m = json.loads(METRICS_PATH.read_text())
        record["deployed"] = {"model": m["selected_model"], "version": m["model_version"], "test": m["test"]}

    PIPELINE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = PIPELINE_RUNS_DIR / f"run_{started.strftime('%Y%m%dT%H%M%SZ')}.json"
    out.write_text(json.dumps(record, indent=2))
    (PIPELINE_RUNS_DIR / "latest.json").write_text(json.dumps(record, indent=2))
    print(f"\nPipeline {record['status']}. Record: {out.relative_to(ROOT)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
