"""
run_lean_judge.py
=================
CLI entry-point for running the Lean Mathlib judge over a GAUSS-Eval dataset.

The judge extracts Lean 4 code from each answer, verifies it via a lean4web-
compatible server (or a local Lean binary), and saves one JSON file per item
together with a summary.

Usage examples
--------------
# Verify all IMO 2025 answers using the lean4web server:
python src/run_lean_judge.py --dataset IMO2025

# Use a local Lean installation instead:
python src/run_lean_judge.py --dataset IMO2025 --backend local --lean-executable lean

# Point at a custom lean4web server:
python src/run_lean_judge.py \\
    --dataset USAMO2025 \\
    --server-url https://my-lean-server/api/compile \\
    --lean-version "leanprover/lean4:v4.15.0" \\
    --timeout 180

# Process only a subset of items:
python src/run_lean_judge.py --dataset DEBUG --sample-indices "0,1,5-10"

# Resume a previous run:
python src/run_lean_judge.py --dataset IMO2025 --resume-from results/IMO2025_lean_1234567890
"""

from __future__ import annotations

import argparse
import asyncio
import glob
import json
import logging
import os
import sys
import time
from multiprocessing import Pool, current_process
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from tqdm import tqdm

# Ensure project root is in path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from data import get_dataset_fn
from src.lean_judge import (
    LEAN4WEB_API_URL,
    DEFAULT_LEAN_VERSION,
    DEFAULT_TIMEOUT,
    LeanJudge,
    run_lean_judge_on_item,
)


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(save_dir: Optional[str] = None, process_id: Optional[str] = None):
    logger_name = f"{__name__}.{process_id}" if process_id else __name__
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(processName)-12s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        log_name = f"run_process_{process_id}.log" if process_id else "run.log"
        fh = logging.FileHandler(
            os.path.join(save_dir, log_name), mode="a", encoding="utf-8"
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(
            logging.Formatter(
                "%(asctime)s │ %(levelname)-8s │ %(processName)-12s │ %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_sample_indices(indices_string: str) -> List[int]:
    """Parse ``"0,1,5-10"`` → ``[0, 1, 5, 6, 7, 8, 9, 10]``."""
    indices: List[int] = []
    for part in indices_string.split(","):
        part = part.strip()
        if "-" in part:
            try:
                start, end = (int(x) for x in part.split("-", 1))
                if start <= end:
                    indices.extend(range(start, end + 1))
            except ValueError:
                print(f"Warning: invalid range '{part}', skipping")
        else:
            try:
                indices.append(int(part))
            except ValueError:
                print(f"Warning: invalid index '{part}', skipping")
    return sorted(set(indices))


def parse_arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Lean Mathlib judge on a GAUSS-Eval dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Dataset
    p.add_argument("--dataset", default="IMO2025",
                   choices=list(get_dataset_fn.keys()),
                   help="Dataset to evaluate.")
    # Lean backend
    p.add_argument("--backend", default="server", choices=["server", "local"],
                   help="Lean verification backend.")
    p.add_argument("--server-url", default=LEAN4WEB_API_URL,
                   help="lean4web-compatible server URL (used with --backend server).")
    p.add_argument("--lean-version", default=DEFAULT_LEAN_VERSION,
                   help="Lean version tag forwarded to the lean4web server.")
    p.add_argument("--lean-executable", default="lean",
                   help="Path to the local `lean` binary (used with --backend local).")
    p.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                   help="Verification timeout in seconds per item.")
    # Run control
    p.add_argument("--sample-indices", default=None,
                   help="Comma-separated indices / ranges to process, e.g. '0,1,5-10'.")
    p.add_argument("--resume-from", default=None,
                   help="Directory of a previous run to resume.")
    p.add_argument("--num-processes", type=int, default=1,
                   help="Number of parallel worker processes.")
    p.add_argument("--save-dir", default="results",
                   help="Root directory for saving results.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Worker function (multiprocessing-safe)
# ---------------------------------------------------------------------------

def _worker(args_tuple) -> Dict[str, Any]:
    """Process one dataset item.  Designed to be called by multiprocessing.Pool."""
    (
        data_item,
        dataset_name,
        save_dir,
        backend,
        server_url,
        lean_version,
        lean_executable,
        timeout,
    ) = args_tuple

    process_name = current_process().name
    logger = setup_logging(save_dir, process_name)

    item_id = data_item.get("id", "?")
    logger.info("Processing item id=%s", item_id)

    try:
        result = run_lean_judge_on_item(
            data_item,
            backend=backend,
            server_url=server_url,
            lean_version=lean_version,
            lean_executable=lean_executable,
            timeout=timeout,
        )

        out_path = os.path.join(save_dir, f"{item_id}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        logger.debug("Saved result to %s", out_path)
        return {"success": True, "id": item_id, "result": result}

    except Exception as exc:
        logger.error("Failed to process item id=%s: %s", item_id, exc, exc_info=True)
        return {"success": False, "id": item_id, "error": str(exc)}


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def get_completed_ids(save_dir: str) -> Set[Any]:
    completed: Set[Any] = set()
    for json_file in glob.glob(os.path.join(save_dir, "*.json")):
        if os.path.basename(json_file) in {"summary.json", "merged_samples.json"}:
            continue
        try:
            with open(json_file, "r") as f:
                data = json.load(f)
            if "id" in data:
                completed.add(data["id"])
        except Exception:
            pass
    return completed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_arguments()

    # ---- Set up save directory ----
    if args.resume_from:
        save_dir = args.resume_from
        if not os.path.exists(save_dir):
            sys.exit(f"Resume directory does not exist: {save_dir}")
        run_name = os.path.basename(save_dir)
        resuming = True
    else:
        run_name = f"{args.dataset}_lean_{int(time.time())}"
        save_dir = os.path.join(args.save_dir, run_name)
        resuming = False

    logger = setup_logging(save_dir)

    logger.info("=" * 60)
    logger.info("%s run: %s", "Resuming" if resuming else "Starting", run_name)
    logger.info("=" * 60)
    logger.info("Dataset         : %s", args.dataset)
    logger.info("Backend         : %s", args.backend)
    if args.backend == "server":
        logger.info("Server URL      : %s", args.server_url)
        logger.info("Lean version    : %s", args.lean_version)
    else:
        logger.info("Lean executable : %s", args.lean_executable)
    logger.info("Timeout         : %ds", args.timeout)
    logger.info("Workers         : %d", args.num_processes)
    logger.info("Save dir        : %s", save_dir)
    logger.info("=" * 60)

    # ---- Load dataset ----
    dataset = get_dataset_fn[args.dataset]
    logger.info("Loaded dataset with %d examples", len(dataset.data))

    # ---- Apply sample-indices filter ----
    if args.sample_indices:
        indices = parse_sample_indices(args.sample_indices)
        available = {item["id"] for item in dataset.data}
        dataset.data = [
            item for item in dataset.data if item["id"] in set(indices) & available
        ]
        logger.info("Filtered to %d examples by sample indices", len(dataset.data))
        if not dataset.data:
            logger.error("No valid samples after filtering.")
            return

    # ---- Resume: skip already-completed items ----
    if resuming:
        done = get_completed_ids(save_dir)
        logger.info("Already completed: %d items", len(done))
        remaining = [item for item in dataset.data if item["id"] not in done]
    else:
        done = set()
        remaining = dataset.data

    logger.info("Items to process: %d", len(remaining))
    if not remaining:
        logger.info("Nothing to do.")
        return

    # ---- Build worker argument tuples ----
    worker_args = [
        (
            item,
            args.dataset,
            save_dir,
            args.backend,
            args.server_url,
            args.lean_version,
            args.lean_executable,
            args.timeout,
        )
        for item in remaining
    ]

    # ---- Execute ----
    if args.num_processes > 1:
        logger.info("Running in parallel with %d workers", args.num_processes)
        with Pool(processes=args.num_processes) as pool:
            results = list(
                tqdm(
                    pool.imap(_worker, worker_args),
                    total=len(worker_args),
                    desc="Lean judging",
                )
            )
    else:
        logger.info("Running sequentially")
        results = []
        for wa in tqdm(worker_args, desc="Lean judging"):
            results.append(_worker(wa))

    # ---- Statistics ----
    successful = sum(1 for r in results if r["success"])
    failed = len(results) - successful

    # Count how many answers actually contained Lean code
    lean_found = sum(
        1 for r in results
        if r["success"] and r["result"].get("lean_code_found", False)
    )
    verified = sum(
        1 for r in results
        if r["success"]
        and r["result"].get("lean_judge_result", {}).get("success", False)
    )

    logger.info("=" * 60)
    logger.info("Run complete")
    logger.info("  Total items processed : %d", len(results))
    logger.info("  Worker successes      : %d", successful)
    logger.info("  Worker failures       : %d", failed)
    logger.info("  Lean code found       : %d", lean_found)
    logger.info("  Lean proofs verified  : %d", verified)
    logger.info("=" * 60)

    # ---- Save summary ----
    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "backend": args.backend,
        "server_url": args.server_url if args.backend == "server" else None,
        "lean_version": args.lean_version if args.backend == "server" else None,
        "lean_executable": args.lean_executable if args.backend == "local" else None,
        "timeout": args.timeout,
        "num_processes": args.num_processes,
        "total_examples": len(dataset.data),
        "newly_processed": len(results),
        "previously_completed": len(done),
        "worker_successes": successful,
        "worker_failures": failed,
        "lean_code_found": lean_found,
        "lean_proofs_verified": verified,
        "sample_indices": args.sample_indices or "all",
        "resumed": resuming,
        "results": results,
    }

    summary_path = os.path.join(save_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    logger.info("Summary saved to: %s", summary_path)


if __name__ == "__main__":
    main()
