"""BF-1.4 — cross-seam integration smoke test.

One test that traverses every BLA-Forge software seam built so far:

  1. BF-0.1 mock_calibration   → CalibrationBundle
  2. BF-0.2 mock_fiducials     → FiducialDetection list
  3. BF-1.0 DemoCollector      → DemoRecord
  4. BF-0.4 save_demo          → demo .json
  5. BF-1.0 DemoBank.add       → demo in-bank
  6. BF-1.0 DemoBank.from_directory → fresh reload
  7. BF-1.1 build_mock_deployment_loop → EpisodeRecord using bank
  8. BF-1.2 cube_xy_displacement_score → outcome scorer
  9. BF-1.2 compute_outcome    → outcome dict
 10. BF-0.4 save_episode/load_episode → EpisodeRecord round-trip
 11. BF-1.2 episode_to_demo   → ep → DemoRecord conversion
 12. BF-1.0 DemoBank.add (2nd) → augmented bank
 13. BF-1.1 deployment loop on augmented bank → retrieves the new demo

If any seam drifts (schema version, dtype, key shape, init_state shape,
fiducial-id encoding), this test catches it before the rest of the
suite. It's the closest thing we have to "would this work on real
hardware" pre-hardware.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from bla.forge import (
    DemoBank,
    DemoCollector,
    build_mock_deployment_loop,
    compute_outcome,
    cube_xy_displacement_score,
    episode_to_demo,
    load_episode,
    mock_calibration,
    mock_fiducials,
    save_demo,
    save_episode,
)


def _key_cube_xy(scene: dict) -> np.ndarray:
    """Retrieval key = cube's world xy. Used at both indexing and
    query time so the bank lives in the same feature space as the
    deploy-side scene."""
    return np.asarray(scene[0], dtype=np.float32)


def _make_demo_via_collector(
    *, demo_id: int, cube_xy: tuple[float, float],
    bundle, n_actions: int = 15, outcome: float = 0.85,
):
    """Build a DemoRecord using the full DemoCollector lifecycle —
    NOT build_mock_demo, since this test wants to exercise the
    operator-side collection path end-to-end."""
    fids = mock_fiducials(
        ids=(0, 1),
        world_xy_per_id={0: tuple(map(float, cube_xy)),
                          1: (0.10, -0.10)},
        bundle=bundle,
    )
    coll = DemoCollector(demo_id=demo_id, task="pickplace", action_dim=7)
    coll.start(initial_fiducials=fids, bundle=bundle,
                  extra={"gripper_open": 1.0})
    rng = np.random.RandomState(demo_id)
    for t in range(n_actions):
        action = rng.uniform(-0.02, 0.02, size=7).astype(np.float32)
        action[6] = 1.0 if t > n_actions // 2 else 0.0
        coll.append_action(action)
    return coll.finalize(
        achieved_outcome=outcome,
        collector_notes=f"collected via DemoCollector, demo_id={demo_id}",
    )


def test_full_pipeline_collect_save_reload_deploy_score_convert_extend(
    tmp_path: Path,
):
    bundle = mock_calibration()
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()

    # ----- seams 1-5: collect 3 demos, save, add to bank -----
    bank = DemoBank.from_directory(bank_dir)
    for did, cube in enumerate([(-0.08, 0.0), (0.00, 0.0), (+0.08, 0.0)]):
        demo = _make_demo_via_collector(
            demo_id=did, cube_xy=cube, bundle=bundle, n_actions=12,
            outcome=0.7 + 0.05 * did,
        )
        path = bank.add(demo)
        assert path.exists()
    assert len(bank) == 3

    # ----- seam 6: fresh reload from disk -----
    bank_reloaded = DemoBank.from_directory(bank_dir)
    assert len(bank_reloaded) == 3
    # The reloaded records are equal to the originals (modulo file I/O)
    for original, reloaded in zip(bank.records, bank_reloaded.records):
        assert original.demo_id == reloaded.demo_id
        assert original.task == reloaded.task
        assert original.achieved_outcome == reloaded.achieved_outcome
        np.testing.assert_allclose(original.actions, reloaded.actions,
                                           atol=1e-6)
        # initial_state must round-trip
        assert original.initial_state["fiducials"]["0"] == \
            reloaded.initial_state["fiducials"]["0"]

    # ----- seam 7: deployment loop with the reloaded bank -----
    # Query at (+0.07, 0.0) — closest to demo_id=2 (cube at +0.08)
    rec = build_mock_deployment_loop(
        bank=bank_reloaded, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.07, 0.0), 1: (0.10, -0.10)},
    )
    assert rec.retrieved_demo["demo_id"] == 2
    assert rec.frames.shape[0] == 12   # demo 2 has 12 actions

    # ----- seams 8-9: scorer + outcome dict -----
    s = cube_xy_displacement_score(rec, slot_id=0, target_m=0.10)
    out = compute_outcome(rec, score_fn=cube_xy_displacement_score,
                              metric_name="cube_disp_m",
                              success_threshold=0.5)
    # The deployment loop's cube doesn't move much (small drift only)
    # so the score is low. That's fine — we just need a consistent value.
    assert 0.0 <= s <= 1.0
    assert out["improvement"] == s
    assert out["metric_name"] == "cube_disp_m"

    # ----- seam 10: episode → disk → reload -----
    ep_path = tmp_path / "ep_deploy.json"
    save_episode(rec, ep_path, use_sidecar=True)
    loaded = load_episode(ep_path)
    np.testing.assert_allclose(loaded.gantry_actions, rec.gantry_actions,
                                       atol=1e-6)
    np.testing.assert_allclose(loaded.decoded_positions, rec.decoded_positions,
                                       atol=1e-6)
    assert loaded.retrieved_demo == rec.retrieved_demo
    assert loaded.router_decision == rec.router_decision

    # ----- seam 11: ep → DemoRecord conversion -----
    # Pretend the operator labelled this episode "good demo" → bring it
    # back into the bank.
    new_demo = episode_to_demo(
        loaded, demo_id=bank.next_demo_id(),
        score_fn=cube_xy_displacement_score,
        collector_notes="recycled from deploy ep_deploy.json",
    )
    assert new_demo.actions.shape == loaded.gantry_actions.shape
    np.testing.assert_array_equal(new_demo.actions, loaded.gantry_actions)
    # Initial state comes from the first decoded step — slot 0 should
    # be near (0.07, 0.0).
    np.testing.assert_allclose(
        new_demo.initial_state["fiducials"]["0"][:2], [0.07, 0.0], atol=5e-3)

    # ----- seam 12: add the new demo to the bank -----
    bank.add(new_demo)
    assert len(bank) == 4

    # ----- seam 13: redeploy — the new demo should now be retrievable -----
    # Query at (+0.07, 0.0), exactly where the new demo lives → it should
    # win retrieval (nn_distance ~ 0 instead of ~0.01 for the old demo 2).
    bank_v2 = DemoBank.from_directory(bank_dir)
    assert len(bank_v2) == 4
    rec2 = build_mock_deployment_loop(
        bank=bank_v2, key_fn=_key_cube_xy,
        query_world_xy_per_id={0: (0.07, 0.0), 1: (0.10, -0.10)},
    )
    assert rec2.retrieved_demo["demo_id"] == new_demo.demo_id
    assert rec2.retrieved_demo["nn_distance"] < 0.005   # very tight match


def test_full_pipeline_no_schema_drift_across_disk(tmp_path: Path):
    """A narrower test that focuses on schema stability: every record
    that round-trips through disk MUST come back with the canonical
    schema versions (no silent migration, no warning)."""
    import warnings
    bundle = mock_calibration()
    bank_dir = tmp_path / "bank"
    bank_dir.mkdir()

    bank = DemoBank.from_directory(bank_dir)
    bank.add(_make_demo_via_collector(
        demo_id=0, cube_xy=(0.0, 0.0), bundle=bundle,
    ))

    # Reload — no UserWarning should fire (matching schema version)
    with warnings.catch_warnings():
        warnings.simplefilter("error")   # promote warnings to errors
        bank_reloaded = DemoBank.from_directory(bank_dir)
        rec = build_mock_deployment_loop(
            bank=bank_reloaded, key_fn=_key_cube_xy,
            query_world_xy_per_id={0: (0.0, 0.0), 1: (0.10, -0.10)},
        )
        ep_path = tmp_path / "ep.json"
        save_episode(rec, ep_path, use_sidecar=True)
        load_episode(ep_path)
