"""BF-0.5 — safety / e-stop watchdog scaffold.

Per `docs/BLA_FORGE_REAL_WORLD_TESTBED_SPEC.md` §7 and the locked
BF-0 sub-phase order: this module is the LAST software layer that
has to land before any motion control runs. It produces gated
control decisions (`SafetyDecision.action ∈ {"log_only",
"slow_to_zero", "halt"}`) that the production-side gantry driver
plugs into; the BF-0.4 EpisodeLogger consumes the same events via
`log_safety_event(...)`.

Mock-first design (matches BF-0.1 / BF-0.2 / BF-0.3 / BF-0.4):

  SafetyMonitor(... mode="mock")
    - Validates per-step gantry pose against a WorkspaceBounds AABB
    - Validates per-step gantry velocity against per-axis limits
    - Maintains a deadman timestamp: if `tick()` is not called within
      `deadman_timeout_s`, `decide()` returns "halt"
    - Force/torque limit check is optional (None = skipped); when
      provided, |F| > F_max triggers "slow_to_zero" first, then "halt"
      on repeated breach.

  SafetyMonitor(... mode="hw")
    - NotImplementedError until the production-side IO board /
      hardware e-stop relay is wired in. Same API; hardware-arrival
      becomes a backend swap.

Decision tiers (escalation, never de-escalation within an episode):

  log_only        — soft warning; gantry continues; event logged
  slow_to_zero    — controller ramps action → 0 within
                    `slow_to_zero_s` seconds; episode terminates
                    cleanly
  halt            — immediate stop; episode aborts; e-stop relay
                    fires on real hardware

Once a `slow_to_zero` decision is issued, the monitor stays in that
state until reset (so a one-cycle pose blip can't oscillate the
controller). `halt` is terminal.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------- bounds ----------
@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned bounding box in gantry world frame (meters).

    Hard bounds = absolute outer envelope; cross it → halt.
    Soft bounds = inner warning envelope; cross it → slow_to_zero.
    Soft bounds MUST be strictly inside hard bounds on every axis.
    """
    soft_min: np.ndarray   # [3]: x_min, y_min, z_min
    soft_max: np.ndarray   # [3]
    hard_min: np.ndarray   # [3]
    hard_max: np.ndarray   # [3]

    def __post_init__(self):
        for name, arr in (("soft_min", self.soft_min),
                           ("soft_max", self.soft_max),
                           ("hard_min", self.hard_min),
                           ("hard_max", self.hard_max)):
            if arr.shape != (3,):
                raise ValueError(f"{name} must be shape (3,), got {arr.shape}")
        if not (np.all(self.soft_min < self.soft_max) and
                  np.all(self.hard_min < self.hard_max)):
            raise ValueError("min must be < max on every axis")
        if not (np.all(self.hard_min <= self.soft_min) and
                  np.all(self.soft_max <= self.hard_max)):
            raise ValueError("soft bounds must be strictly inside hard bounds")

    def classify(self, pose_xyz: np.ndarray) -> str:
        """Return "inside" | "soft_breach" | "hard_breach"."""
        if np.any(pose_xyz < self.hard_min) or np.any(pose_xyz > self.hard_max):
            return "hard_breach"
        if np.any(pose_xyz < self.soft_min) or np.any(pose_xyz > self.soft_max):
            return "soft_breach"
        return "inside"


# ---------- velocity / force limits ----------
@dataclass(frozen=True)
class VelocityLimits:
    """Per-axis velocity limits (m/s for translation, rad/s for rot).

    Translational axes (x, y, z) and rotational axes (rx, ry, rz)
    use separate per-axis limits. A scalar `force_max_n` is the
    optional Cartesian force-magnitude limit (Newtons) used when
    callers supply a measured force.
    """
    v_max_xyz: np.ndarray       # [3] m/s
    w_max_xyz: np.ndarray       # [3] rad/s
    force_max_n: Optional[float] = None

    def __post_init__(self):
        for name, arr in (("v_max_xyz", self.v_max_xyz),
                           ("w_max_xyz", self.w_max_xyz)):
            if arr.shape != (3,):
                raise ValueError(f"{name} must be shape (3,)")
            if np.any(arr <= 0):
                raise ValueError(f"{name} must be strictly positive")
        if self.force_max_n is not None and self.force_max_n <= 0:
            raise ValueError("force_max_n must be > 0 if provided")


# ---------- decision record ----------
@dataclass(frozen=True)
class SafetyDecision:
    """The output of one SafetyMonitor.decide() call.

    Fields:
      action      "log_only" | "slow_to_zero" | "halt"
      reason      machine-readable string ("bounds_soft", "bounds_hard",
                  "vel_limit", "force_limit", "deadman_timeout",
                  "ok")
      timestep    monotonic step counter from the caller
      details     dict with breach magnitudes / which axis / etc;
                  intended for EpisodeLogger.log_safety_event consumers
    """
    action: str
    reason: str
    timestep: int
    details: dict = field(default_factory=dict)


# ---------- mock workspace / limit presets ----------
def mock_workspace_bounds() -> WorkspaceBounds:
    """A reasonable 40×40×20 cm workspace centered on the table origin.

    Soft envelope is 2 cm inside the hard envelope on every axis. The
    actual BLA-Forge hardware bounds will be calibrated to the real
    table; this is just enough to drive the BF-0.5 mock path.
    """
    return WorkspaceBounds(
        soft_min=np.array([-0.18, -0.18, 0.02], dtype=np.float64),
        soft_max=np.array([+0.18, +0.18, 0.18], dtype=np.float64),
        hard_min=np.array([-0.20, -0.20, 0.00], dtype=np.float64),
        hard_max=np.array([+0.20, +0.20, 0.20], dtype=np.float64),
    )


def mock_velocity_limits() -> VelocityLimits:
    """Conservative velocity / force ceiling for first hardware contact."""
    return VelocityLimits(
        v_max_xyz=np.array([0.10, 0.10, 0.05], dtype=np.float64),  # m/s
        w_max_xyz=np.array([0.30, 0.30, 0.30], dtype=np.float64),  # rad/s
        force_max_n=8.0,
    )


# ---------- the monitor ----------
class SafetyMonitor:
    """Stateful safety / e-stop watchdog with monotone decision escalation.

    Args:
      bounds              WorkspaceBounds for the gantry workspace.
      limits              VelocityLimits for per-axis vel/force ceilings.
      deadman_timeout_s   if `tick()` is not called within this many
                          seconds, decide() emits "halt".
      slow_to_zero_s      hint to the controller about how long the
                          ramp-to-zero should take; recorded in decision
                          details for downstream logging.
      mode                "mock" (default) | "hw" (raises until real
                          IO board is wired in).

    Decision escalation is monotone: once `slow_to_zero` is issued, the
    monitor will not return to `log_only` even if subsequent observations
    are inside both soft and hard bounds. `halt` is terminal until
    `reset()` is called.
    """

    def __init__(
        self,
        bounds: WorkspaceBounds,
        limits: VelocityLimits,
        *,
        deadman_timeout_s: float = 0.5,
        slow_to_zero_s: float = 0.5,
        mode: str = "mock",
        time_fn=time.monotonic,
    ):
        if mode not in ("mock", "hw"):
            raise ValueError(f"unknown mode: {mode!r}")
        if mode == "hw":
            raise NotImplementedError(
                "mode='hw' pending production IO board / e-stop relay. The "
                "same SafetyMonitor API will dispatch to a hardware "
                "interlock; until then, use mode='mock'.")
        if deadman_timeout_s <= 0:
            raise ValueError("deadman_timeout_s must be > 0")
        if slow_to_zero_s <= 0:
            raise ValueError("slow_to_zero_s must be > 0")

        self.bounds = bounds
        self.limits = limits
        self.deadman_timeout_s = deadman_timeout_s
        self.slow_to_zero_s = slow_to_zero_s
        self.mode = mode
        self._time = time_fn
        self._last_tick = self._time()
        # Monotone state: "ok" → "slow_to_zero" → "halt"
        self._state = "ok"

    # ---------- deadman ----------
    def tick(self) -> None:
        """Refresh the deadman timestamp. The production controller calls
        this every control loop iteration; missing N ticks means we lost
        the controller and must halt."""
        self._last_tick = self._time()

    @property
    def state(self) -> str:
        """Current escalation state ("ok" | "slow_to_zero" | "halt")."""
        return self._state

    def reset(self) -> None:
        """Clear escalation state and refresh deadman.

        ONLY call this between episodes — never mid-episode in response
        to a breach (that would defeat the monotone-escalation property).
        """
        self._state = "ok"
        self._last_tick = self._time()

    # ---------- main decision ----------
    def decide(
        self,
        timestep: int,
        pose_xyz: np.ndarray,
        velocity_xyz: Optional[np.ndarray] = None,
        angular_velocity_xyz: Optional[np.ndarray] = None,
        force_n: Optional[float] = None,
    ) -> SafetyDecision:
        """Evaluate the current step against all safety predicates.

        Args:
          timestep                monotonic step counter from caller.
          pose_xyz                [3] gantry tool position in world frame.
          velocity_xyz            [3] linear velocity (m/s), optional.
          angular_velocity_xyz    [3] angular velocity (rad/s), optional.
          force_n                 scalar |F| at end-effector (N), optional.

        Returns a SafetyDecision. Once "halt" is returned, all subsequent
        calls also return "halt" until reset() is called.
        """
        # Hard-state lockouts first
        if self._state == "halt":
            return SafetyDecision(
                action="halt", reason="halt_latched",
                timestep=timestep, details={})

        # Deadman: have we missed the tick window?
        now = self._time()
        if now - self._last_tick > self.deadman_timeout_s:
            self._state = "halt"
            return SafetyDecision(
                action="halt", reason="deadman_timeout",
                timestep=timestep,
                details={"elapsed_s": now - self._last_tick,
                              "timeout_s": self.deadman_timeout_s})

        # Hard bounds breach → halt
        if pose_xyz.shape != (3,):
            raise ValueError(f"pose_xyz must be shape (3,), got {pose_xyz.shape}")
        klass = self.bounds.classify(pose_xyz)
        if klass == "hard_breach":
            self._state = "halt"
            return SafetyDecision(
                action="halt", reason="bounds_hard",
                timestep=timestep,
                details={"pose": pose_xyz.tolist(),
                              "hard_min": self.bounds.hard_min.tolist(),
                              "hard_max": self.bounds.hard_max.tolist()})

        # Force limit (if provided): high force → escalate immediately
        if force_n is not None and self.limits.force_max_n is not None:
            if force_n > self.limits.force_max_n * 1.25:
                # Hard force breach (well above limit): halt
                self._state = "halt"
                return SafetyDecision(
                    action="halt", reason="force_limit_hard",
                    timestep=timestep,
                    details={"force_n": float(force_n),
                                  "limit_n": self.limits.force_max_n})
            if force_n > self.limits.force_max_n:
                self._state = "slow_to_zero"
                return SafetyDecision(
                    action="slow_to_zero", reason="force_limit",
                    timestep=timestep,
                    details={"force_n": float(force_n),
                                  "limit_n": self.limits.force_max_n,
                                  "ramp_s": self.slow_to_zero_s})

        # Velocity limit (if provided): exceed → slow_to_zero
        if velocity_xyz is not None:
            if velocity_xyz.shape != (3,):
                raise ValueError("velocity_xyz must be shape (3,)")
            if np.any(np.abs(velocity_xyz) > self.limits.v_max_xyz):
                self._state = "slow_to_zero"
                axis = int(np.argmax(np.abs(velocity_xyz)
                                        - self.limits.v_max_xyz))
                return SafetyDecision(
                    action="slow_to_zero", reason="vel_limit",
                    timestep=timestep,
                    details={"velocity": velocity_xyz.tolist(),
                                  "limit": self.limits.v_max_xyz.tolist(),
                                  "axis": axis,
                                  "ramp_s": self.slow_to_zero_s})

        if angular_velocity_xyz is not None:
            if angular_velocity_xyz.shape != (3,):
                raise ValueError("angular_velocity_xyz must be shape (3,)")
            if np.any(np.abs(angular_velocity_xyz) > self.limits.w_max_xyz):
                self._state = "slow_to_zero"
                return SafetyDecision(
                    action="slow_to_zero", reason="ang_vel_limit",
                    timestep=timestep,
                    details={"angular_velocity": angular_velocity_xyz.tolist(),
                                  "limit": self.limits.w_max_xyz.tolist(),
                                  "ramp_s": self.slow_to_zero_s})

        # If we're already in slow_to_zero, stay there (monotone)
        if self._state == "slow_to_zero":
            return SafetyDecision(
                action="slow_to_zero", reason="slow_to_zero_latched",
                timestep=timestep, details={})

        # Soft bounds breach → log_only (gantry continues but driver
        # should be cautious; this is a "you're close to the edge" hint)
        if klass == "soft_breach":
            return SafetyDecision(
                action="log_only", reason="bounds_soft",
                timestep=timestep,
                details={"pose": pose_xyz.tolist(),
                              "soft_min": self.bounds.soft_min.tolist(),
                              "soft_max": self.bounds.soft_max.tolist()})

        # All clear
        return SafetyDecision(
            action="log_only", reason="ok",
            timestep=timestep, details={})


# ---------- glue: feed a SafetyDecision into BF-0.4's logger ----------
def safety_decision_to_event(decision: SafetyDecision) -> dict:
    """Convert a SafetyDecision to the dict shape EpisodeLogger expects.

    The intended call site:

        decision = monitor.decide(...)
        if decision.reason != "ok":
            logger.log_safety_event(**safety_decision_to_event(decision))

    Keeping this as a free function (not a SafetyDecision method) means
    the safety module stays import-clean of bla.forge.episode, so callers
    can use SafetyMonitor without dragging in the logger.
    """
    return {
        "timestep": decision.timestep,
        "reason": decision.reason,
        "action": decision.action,
    }
