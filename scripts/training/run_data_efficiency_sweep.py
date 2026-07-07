#!/usr/bin/env python
"""Launch the data-efficiency sweep: train_crf.py across (arch x train-fraction).

One training per (arch, fraction) into ``lightning_logs_crf_{arch}_f{pct}_fsst_patient`` — the naming
``data_efficiency_curve.py`` expects. Runs whose checkpoints already exist are skipped, so it is safe to
re-run after an interruption (resumable). Training output streams live.

    pixi run python scripts/training/run_data_efficiency_sweep.py
    pixi run python scripts/training/run_data_efficiency_sweep.py --archs xlstm --fractions 0.1 0.5 1.0
    pixi run python scripts/training/run_data_efficiency_sweep.py --dry-run   # print the commands only

Then build the curve:
    pixi run python scripts/evaluation/data_efficiency_curve.py --archs bilstm xlstm \
        --fractions 0.05 0.1 0.25 0.5 1.0 --folds 3
"""

import argparse
import glob
import subprocess
import sys
import time


def pct_tag(frac: float) -> str:
    return f"{round(frac * 100):02d}"  # 0.05->05, 0.1->10, 1.0->100


def is_done(log_dir: str, folds: int) -> bool:
    """True once every fold has at least one checkpoint (so we can skip a completed run)."""
    return all(glob.glob(f"{log_dir}/fold_{i}/**/*.ckpt", recursive=True) for i in range(folds))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--archs", nargs="+", default=["bilstm", "xlstm"])
    ap.add_argument("--fractions", type=float, nargs="+", default=[0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--downsample", type=int, default=20)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=68)
    ap.add_argument("--template", default="lightning_logs_crf_{arch}_f{pct}_fsst_patient")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running")
    args = ap.parse_args()

    jobs = [(a, f) for a in args.archs for f in args.fractions]
    print(f"Data-efficiency sweep: {len(jobs)} runs = {args.archs} x {args.fractions} @ folds={args.folds}\n")

    failures, t_start = [], time.time()
    for n, (arch, frac) in enumerate(jobs, 1):
        log_dir = args.template.format(arch=arch, pct=pct_tag(frac))
        tag = f"[{n}/{len(jobs)}] {arch} f={frac} -> {log_dir}"

        if is_done(log_dir, args.folds):
            print(f"SKIP (done)  {tag}")
            continue

        cmd = [
            sys.executable, "scripts/training/train_crf.py",
            "--downsample", str(args.downsample),
            "--split-by", "patient",
            "--folds", str(args.folds),
            "--arch", arch,
            "--train-fraction", str(frac),
            "--max-epochs", str(args.max_epochs),
            "--lr", str(args.lr),
            "--seed", str(args.seed),
            "--log-dir", log_dir,
        ]
        if args.dry_run:
            print(f"DRY  {tag}\n     {' '.join(cmd)}")
            continue

        print(f"\n{'=' * 70}\nRUN  {tag}\n{'=' * 70}")
        t0 = time.time()
        result = subprocess.run(cmd)  # noqa: S603 — streams live; args are ours
        dt = time.time() - t0
        if result.returncode != 0:
            print(f"FAILED ({result.returncode}) after {dt:.0f}s: {tag}")
            failures.append(tag)
        else:
            print(f"OK in {dt:.0f}s: {tag}")

    total = time.time() - t_start
    print(f"\n=== sweep finished in {total / 60:.1f} min ===")
    if failures:
        print(f"{len(failures)} FAILED:")
        for f in failures:
            print(f"  {f}")
    elif not args.dry_run:
        print("all runs complete -> now run data_efficiency_curve.py")


if __name__ == "__main__":
    main()
