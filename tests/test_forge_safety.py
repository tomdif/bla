"""Unit tests for bla.forge.safety (BF-0.5).

Verifies the safety / e-stop watchdog scaffold:
  - WorkspaceBounds / VelocityLimits validation
  - SafetyMonitor escalation tiers (log_only → slow_to_zero → halt)
  - Monotone-state guarantee (no de-escalation mid-episode)
  - Deadman timeout
  - Force, velocity, and angular-velocity gates
  - Glue into BF-0.4 EpisodeLogger
"""
from __future__ import annotations

import numpy as np
import pytest

from bla.forge import (
    EpisodeLogger,
    SafetyDecision,
    SafetyMonitor,
    VelocityLimits,
    WorkspaceBounds,
    mock_velocity_limits,
    mock_workspace_bounds,
    safety_decision_to_event,
)


# ---------- bounds ----------
def test_workspace_bounds_shape_validation():
    with pytest.raises(ValueError, match="shape"):
        WorkspaceBounds(
            soft_min=np.zeros(2), soft_max=np.ones(3),
            hard_min=-np.ones(3), hard_max=2*np.ones(3),
        )


def test_workspace_bounds_min_lt_max():
    with pytest.raises(ValueError, match="min must be"):
        WorkspaceBounds(
            soft_min=np.ones(3), soft_max=np.zeros(3),     # inverted
            hard_min=-np.ones(3), hard_max=2*np.ones(3),
        )


def test_workspace_bounds_soft_inside_hard():
    with pytest.raises(ValueError, match="soft bounds must be"):
        WorkspaceBounds(
            soft_min=-2*np.ones(3),   # OUTSIDE hard
            soft_max=2*np.ones(3),
            hard_min=-np.ones(3), hard_max=np.ones(3),
        )


def test_workspace_bounds_classify():
    b = mock_workspace_bounds()
    assert b.classify(np.array([0.0, 0.0, 0.10])) == "inside"
    assert b.classify(np.array([0.19, 0.0, 0.10])) == "soft_breach"
    assert b.classify(np.array([0.25, 0.0, 0.10])) == "hard_breach"


def test_velocity_limits_validate_positive():
    with pytest.raises(ValueError, match="positive"):
        VelocityLimits(
            v_max_xyz=np.array([0.1, -0.1, 0.1]),  # negative
            w_max_xyz=np.array([0.3, 0.3, 0.3]),
        )
    with pytest.raises(ValueError, match="force_max_n must be"):
        VelocityLimits(
            v_max_xyz=np.array([0.1, 0.1, 0.1]),
            w_max_xyz=np.array([0.3, 0.3, 0.3]),
            force_max_n=-5.0,
        )


# ---------- SafetyMonitor: happy path ----------
def _fresh_monitor() -> SafetyMonitor:
    return SafetyMonitor(
        mock_workspace_bounds(),
        mock_velocity_limits(),
        deadman_timeout_s=10.0,   # large so deadman doesn't trip in tests
    )


def test_monitor_ok_inside_bounds():
    m = _fresh_monitor()
    d = m.decide(timestep=0, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d.action == "log_only"
    assert d.reason == "ok"


def test_monitor_soft_breach_logs_only():
    m = _fresh_monitor()
    d = m.decide(timestep=1, pose_xyz=np.array([0.19, 0.0, 0.10]))
    assert d.action == "log_only"
    assert d.reason == "bounds_soft"
    assert m.state == "ok"   # log_only does NOT escalate state


def test_monitor_hard_breach_halts():
    m = _fresh_monitor()
    d = m.decide(timestep=2, pose_xyz=np.array([0.25, 0.0, 0.10]))
    assert d.action == "halt"
    assert d.reason == "bounds_hard"
    assert m.state == "halt"


def test_monitor_halt_is_latched():
    m = _fresh_monitor()
    m.decide(timestep=0, pose_xyz=np.array([0.30, 0.0, 0.10]))  # halt
    # Even with a perfectly safe pose, halt stays
    d = m.decide(timestep=1, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d.action == "halt"
    assert d.reason == "halt_latched"


# ---------- velocity gate ----------
def test_monitor_velocity_limit_slows_to_zero():
    m = _fresh_monitor()
    # x-axis limit is 0.10 m/s; pushing 0.5 m/s should trip
    d = m.decide(
        timestep=0,
        pose_xyz=np.array([0.0, 0.0, 0.10]),
        velocity_xyz=np.array([0.5, 0.0, 0.0]),
    )
    assert d.action == "slow_to_zero"
    assert d.reason == "vel_limit"
    assert d.details["axis"] == 0
    assert m.state == "slow_to_zero"


def test_monitor_angular_velocity_limit_slows_to_zero():
    m = _fresh_monitor()
    d = m.decide(
        timestep=0,
        pose_xyz=np.array([0.0, 0.0, 0.10]),
        angular_velocity_xyz=np.array([2.0, 0.0, 0.0]),  # > 0.30
    )
    assert d.action == "slow_to_zero"
    assert d.reason == "ang_vel_limit"


def test_monitor_slow_to_zero_is_monotone_no_de_escalation():
    """Once slow_to_zero is entered, a clean observation does NOT
    de-escalate to log_only."""
    m = _fresh_monitor()
    m.decide(
        timestep=0, pose_xyz=np.array([0.0, 0.0, 0.10]),
        velocity_xyz=np.array([0.5, 0.0, 0.0]),  # trip
    )
    d2 = m.decide(timestep=1, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d2.action == "slow_to_zero"
    assert d2.reason == "slow_to_zero_latched"
    assert m.state == "slow_to_zero"


# ---------- force gate ----------
def test_monitor_force_above_limit_slows_to_zero():
    m = _fresh_monitor()
    d = m.decide(
        timestep=0,
        pose_xyz=np.array([0.0, 0.0, 0.10]),
        force_n=9.0,   # > 8.0
    )
    assert d.action == "slow_to_zero"
    assert d.reason == "force_limit"


def test_monitor_force_well_above_limit_halts():
    m = _fresh_monitor()
    d = m.decide(
        timestep=0,
        pose_xyz=np.array([0.0, 0.0, 0.10]),
        force_n=12.0,   # > 8.0 * 1.25
    )
    assert d.action == "halt"
    assert d.reason == "force_limit_hard"


# ---------- deadman ----------
def test_monitor_deadman_timeout_halts():
    """Use injectable time_fn so the test doesn't sleep."""
    fake_clock = [0.0]
    m = SafetyMonitor(
        mock_workspace_bounds(),
        mock_velocity_limits(),
        deadman_timeout_s=0.5,
        time_fn=lambda: fake_clock[0],
    )
    # Initial decide is fine
    d = m.decide(timestep=0, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d.action == "log_only"
    # Time passes WITHOUT tick()
    fake_clock[0] = 1.0
    d2 = m.decide(timestep=1, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d2.action == "halt"
    assert d2.reason == "deadman_timeout"


def test_monitor_tick_keeps_deadman_alive():
    fake_clock = [0.0]
    m = SafetyMonitor(
        mock_workspace_bounds(),
        mock_velocity_limits(),
        deadman_timeout_s=0.5,
        time_fn=lambda: fake_clock[0],
    )
    fake_clock[0] = 0.4
    m.tick()
    fake_clock[0] = 0.7   # 0.3 s since last tick (< 0.5 s timeout)
    d = m.decide(timestep=0, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d.action == "log_only"
    assert d.reason == "ok"


# ---------- reset ----------
def test_monitor_reset_clears_escalation():
    m = _fresh_monitor()
    m.decide(timestep=0, pose_xyz=np.array([0.30, 0.0, 0.10]))  # halt
    assert m.state == "halt"
    m.reset()
    assert m.state == "ok"
    # New episode: clean pose returns to ok
    d = m.decide(timestep=0, pose_xyz=np.array([0.0, 0.0, 0.10]))
    assert d.action == "log_only"
    assert d.reason == "ok"


# ---------- hw backend stub ----------
def test_monitor_hw_mode_not_implemented():
    with pytest.raises(NotImplementedError, match="IO board"):
        SafetyMonitor(
            mock_workspace_bounds(), mock_velocity_limits(),
            mode="hw",
        )


def test_monitor_unknown_mode_raises():
    with pytest.raises(ValueError, match="unknown mode"):
        SafetyMonitor(
            mock_workspace_bounds(), mock_velocity_limits(),
            mode="foo",
        )


# ---------- glue into EpisodeLogger ----------
def test_safety_decision_to_event_feeds_logger():
    """A SafetyDecision must convert into a valid log_safety_event call."""
    m = _fresh_monitor()
    logger = EpisodeLogger(ep_id=0, task="pickplace")
    # First step: ok
    d_ok = m.decide(timestep=0, pose_xyz=np.array([0.0, 0.0, 0.10]))
    # Second step: hard breach
    d_halt = m.decide(timestep=1, pose_xyz=np.array([0.30, 0.0, 0.10]))
    # Only log non-ok events (matches the documented call pattern)
    for decision in (d_ok, d_halt):
        if decision.reason != "ok":
            logger.log_safety_event(**safety_decision_to_event(decision))
    logger.set_outcome(success=False, improvement=0.0,
                              metric_name="z_gain", notes="bounds_breach")
    rec = logger.finalize()
    assert len(rec.safety_events) == 1
    ev = rec.safety_events[0]
    assert ev["timestep"] == 1
    assert ev["reason"] == "bounds_hard"
    assert ev["action"] == "halt"


def test_safety_decision_includes_breach_details():
    """The SafetyDecision.details payload is intended for downstream
    analytics; verify it carries enough information to reconstruct the
    breach without re-running the monitor."""
    m = _fresh_monitor()
    d = m.decide(
        timestep=5,
        pose_xyz=np.array([0.0, 0.0, 0.10]),
        velocity_xyz=np.array([0.0, 0.5, 0.0]),
    )
    assert d.details["axis"] == 1
    np.testing.assert_allclose(d.details["limit"], [0.10, 0.10, 0.05])
    np.testing.assert_allclose(d.details["velocity"], [0.0, 0.5, 0.0])
    assert "ramp_s" in d.details


def test_safety_decision_is_frozen():
    """SafetyDecision is a frozen dataclass; downstream code can rely
    on immutable decision records."""
    d = SafetyDecision(action="log_only", reason="ok", timestep=0)
    with pytest.raises(Exception):  # FrozenInstanceError
        d.action = "halt"   # type: ignore[misc]
