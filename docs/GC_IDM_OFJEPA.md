# GC-IDM for OF-JEPA — amortizing planning into a learned inverse map

**Source.** Nguyen, Xu & Huang, *Latent Geometry Beyond Search: Amortizing
Planning in World Models* (2026). On a SIGReg-regularized LeWM latent, a
~1.5M-param goal-conditioned inverse-dynamics MLP `(z_t, z_g, h) → a_t` matches
or beats CEM in **7/8** env-protocol cells at **100–130× lower** per-decision
cost, and the result holds across CEM/MPPI/iCEM/gradient planners and a 500×
compute sweep. Claim: a sufficiently smooth, action-sensitive latent has
*already absorbed* the structure that test-time search recovers.

## Why this is relevant to BLA specifically

1. **The enabling precondition is already met.** The paper's result hinges on
   the latent being smooth/action-sensitive via **SIGReg** (Balestriero & LeCun
   2025). `system1_jepa` already trains with SIGReg ("strictly stronger than
   VICReg"). We are not betting on a regularizer we don't have.

2. **It's the zero-search endpoint of our Phase-18 arc.** We found empirically
   that *less search beats more search around good priors*
   (`search_budget_zero_around_expert_demos`,
   `less_search_when_score_anticorrelated`): `demo_no_cem` ranked #1, light CEM
   barely additive. GC-IDM takes this to the limit — **no search at all**.

3. **Third instance of "reconstruction ≠ task transfer."** The paper's
   Appendix-D pairwise IDM hit **R²=0.993** reconstruction yet planned poorly —
   the bottleneck was constructing a valid latent path, not decoding actions.
   This is our `proxy_vs_end_effect_gate` lesson again, and the BCMI inverse-head
   result again (near-perfect in-distribution reconstruction = memorization,
   ~chance on held-out). High proxy/recon does not imply task transfer. **Gate on
   the end-effect, never the reconstruction R².**

## What's implemented (`system1_jepa/gc_idm.py`)

`GCInverseDynamics(state_dim, goal_dim, action_dim=7, hidden=512, n_layers=3,
mode=...)` — 3-layer residual MLP on `concat(z_t, z_g)` with **AdaLN-Zero**
modulation by a sinusoidally-embedded remaining horizon (zero-init, so horizon
enters only as the loss demands). `train_gc_idm_supervised(...)` regresses to
demo action labels on `(z_t, z_{t+h}, h, a_t)` tuples (offline, no env). `act(...)`
is the one-step closed-loop control law (re-encode each step, one forward, no
rollout). Unit-tested in `tests/test_gc_idm.py`.

Two modes:
- **`flatten`** (paper-faithful): caller flattens OF-JEPA slots (recommend
  `ObjectFileBatch.full_slot`); goal is the goal frame's latent.
- **`perfile`** (OF-JEPA-native): exploits that object files are **identity-
  aligned across frames** (persistent `id_proto`), so `z_t` and `z_g` are
  slot-aligned and a per-file inverse map is well-defined. A shared MLP processes
  each identity-aligned file pair, pooled across files by **Sinkhorn match
  confidence**. This is the genuinely new piece vs the paper — LeWM has a single
  global latent and cannot do this.

## The experiment (against the locked recipe)

Build tuples from the existing Robosuite demo set already used to train the world
model: along each demo, sample `(t, h)` with `h ~ U[1, H_max]`, encode `z_t`,
`z_{t+h}` via the frozen OF-JEPA substrate (`substrate.observe`), label `a_t`.
Train both modes (~20 min/env on one GPU per the paper). Then closed-loop eval
with `act(...)` against:
- **`demo_no_cem`** (current rank-1 prior),
- **locked recipe** = `demo_no_cem` + light CEM (K=32, 1 iter) + value-head
  `combined_sum`,
- **CEM** baseline.

Report success rate, per-decision wall-clock, and the Phase-18 trajectory-quality
metrics (latent monotonicity, action jerk). Reuse the harness in the Phase-18
runners; GC-IDM slots in where `PlanProposalPolicy`/CEM are selected.

## Prediction (stated up front, so a mixed result is still a result)

The paper's **one weak regime is Push-T — contact-rich, long-horizon — where CEM
stays competitive and inverse recovery wanes.** Our suite (Lift, PickPlaceCan,
NutAssemblySquare, ToolHang) is **all contact-rich**, i.e. precisely that regime.
So the honest prediction is:
- competitive-or-winning on the navigation-like / less-contact settings and at
  large horizon budgets where search is wasteful;
- **lagging the locked recipe on the delicate contact phases.**

Either outcome is informative: a win says OF-JEPA's slot geometry is as
amortization-friendly as LeWM's global manifold; a loss localizes exactly where
contact planning still needs search — and whether `perfile` (identity-aligned,
confidence-pooled) closes the gap that `flatten` cannot. Do **not** bank "GC-IDM
replaces search" before the contact-task end-effect clears, per lesson #3 above.
