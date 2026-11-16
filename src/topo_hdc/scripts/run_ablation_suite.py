#!/usr/bin/env python3
"""Run all registered ablations across corruption settings."""

from __future__ import annotations

import argparse
import ast
import csv
from datetime import datetime
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time

from topo_hdc.ablations import ABLATION_PRESETS


CORRUPTIONS = (
    "none",
    "rotation",
    "gaussian",
    "saltpepper",
    "cutout",
    "zoom",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", default="mnist")
    parser.add_argument("--num-classes", type=int, default=10)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1.0)

    parser.add_argument(
        "--ablations",
        nargs="+",
        choices=list(ABLATION_PRESETS),
        default=list(ABLATION_PRESETS),
    )
    parser.add_argument(
        "--corruptions",
        nargs="+",
        choices=CORRUPTIONS,
        default=list(CORRUPTIONS),
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("runs/ablation_suite"),
    )
    parser.add_argument(
        "--command",
        default="topo-hdc-run",
        help="CLI executable, or a command such as 'python -m topo_hdc.cli'.",
    )

    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true")

    parser.add_argument(
        "--force-recompute-test-features",
        action="store_true",
    )
    parser.add_argument(
        "--force-recompute-features",
        action="store_true",
        help="Expensive: forces all feature splits to recompute in every run.",
    )

    parser.add_argument(
        "--extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help=(
            "Additional topo-hdc-run arguments. Place this option last, "
            "for example: --extra-args --cutout 12"
        ),
    )

    return parser.parse_args()


def extract_metrics(log_path: Path) -> dict:
    """
    Read the final metrics dictionary printed by topo-hdc-run.

    This matches the current CLI output shown in the conversation.
    """
    lines = log_path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    for line in reversed(lines):
        line = line.strip()

        if not line.startswith("{"):
            continue

        try:
            metrics = ast.literal_eval(line)
        except (ValueError, SyntaxError):
            continue

        if (
            isinstance(metrics, dict)
            and "acc_before" in metrics
            and "acc_after" in metrics
        ):
            return metrics

    raise RuntimeError(
        f"No final metrics dictionary found in {log_path}."
    )


def atomic_write_json(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2),
        encoding="utf-8",
    )
    temporary.replace(path)


def csv_value(value):
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def save_summaries(root: Path, records: list[dict]) -> None:
    atomic_write_json(root / "results.json", records)

    preferred_fields = [
        "dataset",
        "ablation",
        "corruption",
        "seed",
        "status",
        "returncode",
        "elapsed_seconds",
        "acc_before",
        "acc_after",
        "acc_before_percent",
        "acc_after_percent",
        "val_acc_before",
        "val_acc_after",
        "hog_acc_before",
        "hog_acc_after",
        "fourier_mellin_acc_before",
        "fourier_mellin_acc_after",
        "pi_acc_before",
        "pi_acc_after",
        "weights_before",
        "weights_after",
        "onlinehd_applied",
        "log_path",
        "out_dir",
        "error",
    ]

    additional_fields = sorted(
        {
            key
            for record in records
            for key in record
        } - set(preferred_fields)
    )
    fields = preferred_fields + additional_fields

    temporary = root / "results.csv.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()

        for record in records:
            writer.writerow({
                key: csv_value(value)
                for key, value in record.items()
            })

    temporary.replace(root / "results.csv")


def make_command(args, ablation, corruption, seed, run_dir):
    command = shlex.split(args.command) + [
        "--dataset", args.dataset,
        "--num-classes", str(args.num_classes),
        "--seed", str(seed),
        "--ablation", ablation,
        "--corrupt", corruption,
        "--epochs", str(args.epochs),
        "--lr", str(args.lr),
        "--out-dir", str(run_dir),
    ]

    if args.force_recompute_features:
        command.append("--force-recompute-features")

    if args.force_recompute_test_features:
        command.append("--force-recompute-test-features")

    command.extend(args.extra_args)
    return command


def execute(command, log_path: Path) -> int:
    """Run sequentially and stream output to both the console and a log."""
    with log_path.open("w", encoding="utf-8") as log:
        log.write("COMMAND: " + shlex.join(command) + "\n\n")
        log.flush()

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )

        try:
            assert process.stdout is not None

            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()

            return process.wait()

        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise


def main() -> int:
    args = parse_args()
    root = args.out_dir.resolve()

    if not args.dry_run:
        root.mkdir(parents=True, exist_ok=True)

    records = []
    total = (
        len(args.ablations)
        * len(args.corruptions)
        * len(args.seeds)
    )

    counter = 0

    for seed in args.seeds:
        for ablation in args.ablations:
            for corruption in args.corruptions:
                counter += 1

                run_dir = (
                    root
                    / args.dataset
                    / ablation
                    / corruption
                    / f"seed_{seed}"
                )
                log_path = run_dir / "run.log"
                result_path = run_dir / "suite_result.json"

                command = make_command(
                    args, ablation, corruption, seed, run_dir
                )

                print(
                    f"\n[{counter}/{total}] "
                    f"{ablation} | {corruption} | seed={seed}"
                )
                print(shlex.join(command))

                if args.dry_run:
                    continue

                run_dir.mkdir(parents=True, exist_ok=True)

                # Resume only a successful run with the exact same command.
                if args.resume and result_path.is_file():
                    try:
                        saved = json.loads(
                            result_path.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        saved = None

                    if (
                        saved is not None
                        and saved.get("status") == "success"
                        and saved.get("command") == command
                    ):
                        print("Skipping completed run.")
                        records.append(saved)
                        save_summaries(root, records)
                        continue

                record = {
                    "dataset": args.dataset,
                    "ablation": ablation,
                    "corruption": corruption,
                    "seed": seed,
                    "command": command,
                    "started_at": datetime.now().astimezone().isoformat(),
                    "log_path": str(log_path),
                    "suite_run_dir": str(run_dir),
                }

                start = time.perf_counter()

                try:
                    returncode = execute(command, log_path)
                    record["returncode"] = returncode

                    if returncode != 0:
                        raise RuntimeError(
                            f"CLI exited with code {returncode}; "
                            f"see {log_path}."
                        )

                    metrics = extract_metrics(log_path)
                    record.update(metrics)
                    record["status"] = "success"
                    record["acc_before_percent"] = (
                        100.0 * float(metrics["acc_before"])
                    )
                    record["acc_after_percent"] = (
                        100.0 * float(metrics["acc_after"])
                    )

                except KeyboardInterrupt:
                    record["status"] = "interrupted"
                    record["error"] = "Interrupted by user."
                    record["elapsed_seconds"] = (
                        time.perf_counter() - start
                    )
                    atomic_write_json(result_path, record)
                    records.append(record)
                    save_summaries(root, records)
                    return 130

                except Exception as exc:
                    record["status"] = "failed"
                    record["error"] = str(exc)
                    print(f"FAILED: {exc}", file=sys.stderr)

                record["elapsed_seconds"] = time.perf_counter() - start

                atomic_write_json(result_path, record)
                records.append(record)
                save_summaries(root, records)

                if (
                    record["status"] != "success"
                    and args.stop_on_error
                ):
                    return 1

    if args.dry_run:
        print(f"\nPlanned runs: {total}")
        return 0

    succeeded = sum(r["status"] == "success" for r in records)
    failed = len(records) - succeeded

    print(f"\nFinished: {succeeded} succeeded; {failed} failed.")
    print(f"CSV summary: {root / 'results.csv'}")
    print(f"JSON summary: {root / 'results.json'}")

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
