"""BF-1.3 — bla.forge command-line introspection.

Two subcommands:

  python -m bla.forge.cli summarize-bank PATH
    Print a one-page summary of a DemoBank directory: total demo count,
    breakdown by task, outcome distribution, retrieval-key coverage of
    the canonical xy plane.

  python -m bla.forge.cli summarize-episode PATH
    Print a one-page summary of an EpisodeRecord JSON: ep_id, task,
    timestamp, n_steps, outcome, retrieved_demo, router_decision,
    safety_events count, and (if decoded_positions are present) the
    cube_xy_displacement_score for slot 0.

Both subcommands write to stdout in a deterministic, parseable format
(key: value, one per line) so they can be diffed in CI or piped through
grep/awk. No external dependencies — pure-stdlib argparse.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from bla.forge.demo_bank import DemoBank
from bla.forge.episode import load_episode
from bla.forge.outcome import cube_xy_displacement_score


def _print_kv(key: str, value, indent: int = 0) -> None:
    pad = " " * indent
    print(f"{pad}{key}: {value}")


# ---------- summarize-bank ----------
def cmd_summarize_bank(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_dir():
        print(f"error: not a directory: {path}", file=sys.stderr)
        return 2
    bank = DemoBank.from_directory(path)
    _print_kv("directory", str(path.resolve()))
    _print_kv("n_demos", len(bank))
    if len(bank) == 0:
        return 0

    # Task breakdown
    task_counts: dict[str, int] = {}
    for r in bank.records:
        task_counts[r.task] = task_counts.get(r.task, 0) + 1
    print("tasks:")
    for t in sorted(task_counts):
        _print_kv(t, task_counts[t], indent=2)

    # Outcome distribution
    outcomes = np.array(
        [r.achieved_outcome for r in bank.records], dtype=np.float64)
    print("outcome:")
    _print_kv("min", f"{outcomes.min():.4f}", indent=2)
    _print_kv("max", f"{outcomes.max():.4f}", indent=2)
    _print_kv("mean", f"{outcomes.mean():.4f}", indent=2)
    _print_kv("median", f"{float(np.median(outcomes)):.4f}", indent=2)
    _print_kv("std", f"{outcomes.std():.4f}", indent=2)

    # Action stream stats
    n_actions = np.array([r.actions.shape[0] for r in bank.records])
    print("actions_per_demo:")
    _print_kv("min", int(n_actions.min()), indent=2)
    _print_kv("max", int(n_actions.max()), indent=2)
    _print_kv("mean", f"{n_actions.mean():.1f}", indent=2)

    # Retrieval-key coverage: span of fiducial-0 xy
    coverage = _xy_coverage_of_fiducial_0(bank)
    if coverage is not None:
        x_min, x_max, y_min, y_max = coverage
        print("fiducial_0_xy_coverage:")
        _print_kv("x_min", f"{x_min:.4f}", indent=2)
        _print_kv("x_max", f"{x_max:.4f}", indent=2)
        _print_kv("y_min", f"{y_min:.4f}", indent=2)
        _print_kv("y_max", f"{y_max:.4f}", indent=2)
        _print_kv("x_span_m", f"{x_max - x_min:.4f}", indent=2)
        _print_kv("y_span_m", f"{y_max - y_min:.4f}", indent=2)

    # Demo IDs
    ids = sorted(r.demo_id for r in bank.records)
    _print_kv("demo_id_range", f"{ids[0]}..{ids[-1]}")
    _print_kv("next_demo_id", bank.next_demo_id())
    return 0


def _xy_coverage_of_fiducial_0(
    bank: DemoBank,
) -> Optional[tuple[float, float, float, float]]:
    """Return (x_min, x_max, y_min, y_max) for fiducial '0' across the bank,
    or None if no record carries it."""
    xs, ys = [], []
    for r in bank.records:
        fids = r.initial_state.get("fiducials", {})
        if "0" in fids:
            xs.append(fids["0"][0])
            ys.append(fids["0"][1])
    if not xs:
        return None
    return min(xs), max(xs), min(ys), max(ys)


# ---------- summarize-episode ----------
def cmd_summarize_episode(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        return 2
    rec = load_episode(path)
    _print_kv("path", str(path.resolve()))
    _print_kv("ep_id", rec.ep_id)
    _print_kv("task", rec.task)
    _print_kv("timestamp", rec.timestamp)
    _print_kv("schema_version", rec.schema_version)
    _print_kv("n_steps", int(rec.frames.shape[0]))
    print("outcome:")
    for k in ("success", "improvement", "metric_name", "notes"):
        if k in rec.outcome:
            _print_kv(k, rec.outcome[k], indent=2)
    print("router_decision:")
    for k in ("recipe", "rationale"):
        if k in rec.router_decision:
            _print_kv(k, rec.router_decision[k], indent=2)
    print("retrieved_demo:")
    for k in ("demo_id", "nn_distance", "filter_passed"):
        if k in rec.retrieved_demo:
            _print_kv(k, rec.retrieved_demo[k], indent=2)
    _print_kv("n_safety_events", len(rec.safety_events))
    _print_kv("n_perturbations", len(rec.perturbations))

    # Diagnostic: cube_xy_displacement for slot 0
    if rec.decoded_positions.size > 0:
        try:
            s = cube_xy_displacement_score(rec, slot_id=0, target_m=0.10)
            _print_kv("cube_xy_disp_score_slot0", f"{s:.4f}")
        except ValueError:
            pass
    return 0


# ---------- argparse plumbing ----------
def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="bla.forge.cli",
        description="BLA-Forge bank/episode introspection",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_bank = sub.add_parser(
        "summarize-bank",
        help="summarize a DemoBank directory",
    )
    p_bank.add_argument("path", help="path to bank directory")
    p_bank.set_defaults(func=cmd_summarize_bank)

    p_ep = sub.add_parser(
        "summarize-episode",
        help="summarize a single EpisodeRecord JSON",
    )
    p_ep.add_argument("path", help="path to episode .json")
    p_ep.set_defaults(func=cmd_summarize_episode)

    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = make_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
