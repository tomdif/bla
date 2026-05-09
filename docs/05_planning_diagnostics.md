# Pillar 5 — Planning + Diagnostics

**Code:** `system1_jepa/planning.py`, `system1_jepa/navigate_env.py`,
`scripts/phase1f_navigate_plan.py`, `diagnostics/`.

## Why a planning task at all

"Loss went down" is unfalsifiable. The blueprint promises a *world model*
that can be planned against. The only honest test of that promise is to
plug the temporal predictor into a Model-Predictive-Control loop and
measure success rate on a downstream task.

`navigate_env.py::NavigateEnv` is the smallest such task: a bright
patch on a black canvas, a target position drawn each episode, action
= (dx, dy). Reward = -L2 distance to target. Success = within radius
within `max_steps`.

## CEM (`planning.py::cem_plan`)

Standard cross-entropy method. Sample action chunks from a Gaussian,
roll out the temporal predictor, score with the predictor's reward
head (or a custom `reward_fn` over predicted z trajectories), refit
the Gaussian to the top-k elites. Returns the best action chunk
sequence; the agent executes the first chunk and re-plans.

Tunables: `horizon`, `iterations`, `population`, `elite_frac`.
Defaults are conservative; expect to push population to 256+ on real
tasks.

## Diagnostics

Three honest probes:

1. **Linear probe** (`diagnostics/linear_probe.py`) — freeze JEPA
   features, train a single linear layer on a labeled property
   (position, color, motion). High accuracy = encoder learned that
   property.
2. **Diffusion canvas trajectory** (`diagnostics/visualize.py`) —
   decode tokens at every step of the denoising integration. Use to
   see whether early steps are structural and late steps are
   syntactic (the blueprint's claim) or whether everything happens at
   one t.
3. **RAM attention dump** (`diagnostics/ram_attention.py`) — per-head
   weight distribution + entropy. Use to check whether the reader is
   committing (low entropy) or hedging (high entropy near `log(top_k)`).
