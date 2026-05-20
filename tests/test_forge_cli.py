"""Unit tests for bla.forge.cli (BF-1.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from bla.forge import (
    DemoBank,
    build_mock_demo,
    build_mock_deployment_loop,
    build_mock_episode_loop,
    save_demo,
    save_episode,
)
from bla.forge.cli import main


# ---------- summarize-bank ----------
def test_cli_summarize_bank_empty_dir(tmp_path: Path, capsys):
    rc = main(["summarize-bank", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_demos: 0" in out


def test_cli_summarize_bank_with_demos(tmp_path: Path, capsys):
    for did, cube in enumerate([(0.0, 0.0), (0.05, 0.05), (-0.05, 0.05)]):
        save_demo(
            build_mock_demo(demo_id=did, cube_xy=cube,
                              achieved_outcome=0.6 + 0.1 * did),
            tmp_path / f"demo_{did:04d}.json",
        )
    rc = main(["summarize-bank", str(tmp_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "n_demos: 3" in out
    assert "pickplace: 3" in out
    assert "outcome:" in out
    assert "min: 0.6000" in out
    assert "max: 0.8000" in out
    assert "fiducial_0_xy_coverage:" in out
    assert "x_span_m: 0.1000" in out
    assert "next_demo_id: 3" in out


def test_cli_summarize_bank_missing_dir_returns_2(tmp_path: Path, capsys):
    rc = main(["summarize-bank", str(tmp_path / "nope")])
    assert rc == 2
    assert "not a directory" in capsys.readouterr().err


def test_cli_summarize_bank_mixed_tasks(tmp_path: Path, capsys):
    save_demo(build_mock_demo(demo_id=0, task="pickplace"),
                tmp_path / "demo_0000.json")
    save_demo(build_mock_demo(demo_id=1, task="lift"),
                tmp_path / "demo_0001.json")
    save_demo(build_mock_demo(demo_id=2, task="lift"),
                tmp_path / "demo_0002.json")
    main(["summarize-bank", str(tmp_path)])
    out = capsys.readouterr().out
    assert "lift: 2" in out
    assert "pickplace: 1" in out


# ---------- summarize-episode ----------
def test_cli_summarize_episode_basic(tmp_path: Path, capsys):
    rec = build_mock_episode_loop(ep_id=42, n_steps=20, include_safety=False)
    ep_path = tmp_path / "ep0.json"
    save_episode(rec, ep_path, use_sidecar=True)
    rc = main(["summarize-episode", str(ep_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ep_id: 42" in out
    assert "task: pickplace" in out
    assert "n_steps: 20" in out
    assert "recipe: E2_FAST" in out
    assert "success: True" in out
    # Cube swept 0.20m → score saturates at 1.0
    assert "cube_xy_disp_score_slot0: 1.0000" in out


def test_cli_summarize_episode_with_safety_breach(tmp_path: Path, capsys):
    rec = build_mock_episode_loop(
        ep_id=0, n_steps=30, include_safety=True, safety_breach_at_step=5,
    )
    ep_path = tmp_path / "ep.json"
    save_episode(rec, ep_path, use_sidecar=True)
    main(["summarize-episode", str(ep_path)])
    out = capsys.readouterr().out
    assert "n_safety_events: 1" in out
    assert "success: False" in out


def test_cli_summarize_episode_deployment(tmp_path: Path, capsys):
    """An episode produced by build_mock_deployment_loop should
    summarize with a populated retrieved_demo block."""
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()
    for did in range(3):
        save_demo(build_mock_demo(demo_id=did, cube_xy=(0.05*did, 0.0)),
                    bank_dir / f"demo_{did:04d}.json")
    bank = DemoBank.from_directory(bank_dir)
    rec = build_mock_deployment_loop(
        bank=bank, key_fn=lambda s: s[0],
        query_world_xy_per_id={0: (0.10, 0.0), 1: (0.10, -0.10)},
    )
    ep_path = tmp_path / "deploy.json"
    save_episode(rec, ep_path, use_sidecar=True)
    main(["summarize-episode", str(ep_path)])
    out = capsys.readouterr().out
    assert "retrieved_demo:" in out
    assert "demo_id: 2" in out  # closest cube to (0.10, 0.0)


def test_cli_summarize_episode_missing_file_returns_2(tmp_path: Path, capsys):
    rc = main(["summarize-episode", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "not a file" in capsys.readouterr().err


# ---------- argparse plumbing ----------
def test_cli_no_subcommand_exits_nonzero():
    with pytest.raises(SystemExit) as exc:
        main([])
    assert exc.value.code != 0
