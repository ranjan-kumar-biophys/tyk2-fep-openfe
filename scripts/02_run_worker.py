"""Step 2: run transformations with `openfe quickrun`, importing OpenFE only once.

Every `openfe quickrun` call starts a fresh Python that re-imports OpenFE (and PyTorch);
on a busy cluster's network filesystem that took ~17 minutes per import. A worker imports
once, then pulls transformation JSONs from a shared task list until it is empty, calling
quickrun in-process (same output files, so `openfe gather` works unchanged). Start one
worker per GPU; finished legs are skipped, so the job can simply be resubmitted.
  CUDA_VISIBLE_DEVICES=0 python scripts/02_run_worker.py tasks.txt
"""
import argparse
import datetime
import fcntl
from pathlib import Path

from openfecli.commands.quickrun import quickrun

ROOT = Path(__file__).resolve().parents[1]


def next_task(task_file):
    """Claim the next line of task_file; a counter file under flock keeps workers in step."""
    with open(f"{task_file}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        counter = Path(f"{task_file}.claimed")
        n = int(counter.read_text()) if counter.exists() else 0
        lines = [l for l in Path(task_file).read_text().splitlines() if l.strip()]
        counter.write_text(str(n + 1))
    return Path(lines[n]) if n < len(lines) else None


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task_file", help="one transformation JSON path per line")
    p.add_argument("--out", type=Path, default=ROOT / "work" / "results", help="result JSONs")
    p.add_argument("--runs", type=Path, default=ROOT / "work" / "runs", help="per-leg work dirs")
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    while (transformation := next_task(args.task_file)) is not None:
        name = transformation.stem
        out = args.out / f"{name}.json"
        if out.exists() and out.stat().st_size:
            continue
        print(f"{datetime.datetime.now():%F %T} start {name}", flush=True)
        try:
            quickrun.main([str(transformation), "-o", str(out), "-d", str(args.runs / name)],
                          standalone_mode=False)
        except (Exception, SystemExit) as err:  # record and move on to the next leg
            with open(ROOT / "TASK_FAILED", "a") as fh:
                fh.write(f"{name}: {type(err).__name__}: {err}\n")
        print(f"{datetime.datetime.now():%F %T} end   {name}", flush=True)


if __name__ == "__main__":
    main()
