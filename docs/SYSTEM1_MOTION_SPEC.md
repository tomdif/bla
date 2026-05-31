# System-1 Motion Substrate: Specification
## A Substrate Rewrite Designed to Pass Gate 0

This is the design document for the substrate rewrite that the OF-JEPA-navigate failure arc pointed at. The four-fix sequence (proper pretrain → agent grounding → SIGReg → escalating combinations) all came back at chance level on Gate 0 (<5px agent-decode error on held-out frames). The conclusion that finding forces: the substrate's training objective does not have motion-tracking as a target, so no amount of representation-side regularization will make the state expose dynamic position. The substrate is doing what it was trained to do; it was trained for the wrong thing.

This spec is the minimum substrate that should pass Gate 0, written from first principles rather than as a delta against OF-JEPA-navigate. It also includes a few design choices that aren't in the standard JEPA recipe — I'll mark each as "standard recipe," "evidence-driven correction from the four-fix arc," or "speculative, worth testing." The first two earn the design's right to exist. The third is where the project earns the right to claim something interesting.

## 1. What Gate 0 actually demands

Before designing the objective, state the gate precisely. The gate is not "a good representation." It is:

> Given a frozen checkpoint of the trained substrate, a small decoder (≤2-layer MLP) trained on held-out frames must recover the agent's position to <5 pixels mean L2 error on a held-out test set.

Three properties this gate has that matter for the design:

**It's a held-out probe, not a trained-jointly probe.** The decoder is small and trained after the substrate is frozen. The substrate doesn't get gradient signal from the decoder during its own training. This rules out the trivial solution of "make position decodable by giving the substrate a position-decode head and training it" — that fails the spirit of the gate because it doesn't show the substrate would have learned to track position from a more general objective.

**It's a position probe, not a generic feature-quality probe.** The substrate doesn't have to be good for everything. It has to expose dynamic position information in a linearly-or-nearly-linearly-recoverable form. This is a sharper target than "useful representation" and the design should aim at it specifically.

**It's a pass/fail gate, not a leaderboard.** 4.9px is a pass. 5.1px is a fail. There's no partial credit. This forces the design to optimize for the binary threshold, not for marginal improvements that look good but don't cross.

The implication for the design: every component should be traceable to "does this make held-out position decode work, or not." Components that don't have that traceability should be cut or marked as speculative.

## 2. What the four-fix arc actually showed

Spelling out the evidence the design has to honor:

**The OF-JEPA-navigate base substrate failed Gate 0.** Position was not linearly decodable from the state at <5px. The base substrate was trained on an objective that doesn't reward motion tracking, so the state collapsed to slow features (lighting, scene structure, background) that are predictable without tracking the agent.

**Proper pretraining did not fix it.** More compute on the same objective doesn't produce motion tracking. This rules out "we just needed more training." It implicates the objective directly.

**Agent grounding did not fix it.** Attempts to bias the representation toward the agent (presumably some form of agent-region attention or weighted loss) did not flow through to position decodability. This is informative: the bottleneck isn't where the model attends, it's what the loss rewards. You can attend to the agent and still not encode position if the objective doesn't reward encoding position.

**SIGReg did not fix it.** Anti-collapse regularization on the embedding distribution doesn't redirect what kind of information is encoded. SIGReg controls the geometry of the embedding cloud, not its content. This is consistent with the broader lesson from the experimental arc: regularizers control amount and shape of information, not kind.

The structural lesson: **objective changes the kind, regularizers change the amount/shape.** Gate 0 needs a different kind of information in the state (motion tracking), so the fix has to be at the objective level. That's where the spec puts its weight.

## 3. The objective: action-conditioned next-state prediction with two auxiliary signals

The primary objective is action-conditioned next-state prediction in latent space:

$$\mathcal{L}_\text{pred}(\theta, \omega) = \mathbb{E}_{(x_t, a_t, x_{t+1})} \left[ \frac{1}{d_z \sigma_z^2} \left\| T_\omega(f_\theta(x_t), a_t) - \text{sg}[\bar f(x_{t+1})] \right\|_2^2 \right]$$

with $f_\theta$ the encoder, $T_\omega$ the latent dynamics, $\bar f$ the EMA target encoder, and $\text{sg}$ the stop-gradient.

This is the V-JEPA 2-AC pattern, history-conditioning deferred for now to keep the substrate evaluation clean. **Standard recipe.** I'm not claiming anything new about the prediction loss itself. The novelty has to come from what surrounds it, not from this term.

Why this should pass Gate 0 when OF-JEPA-navigate didn't: the loss directly rewards predicting future states from current state + action. The only way to predict accurately is for the state to contain enough information about *what's moving and where* that the dynamics model can simulate the consequence of the action. Position is part of that information. The objective makes position-tracking instrumental to lowering the loss, which is what was missing.

This alone should be sufficient. The auxiliaries are insurance against specific failure modes the four-fix arc surfaced.

### Auxiliary 1: Action-leverage floor (evidence-driven correction)

The action_leverage diagnostic from the experimental arc work showed that on smooth video the trivial baseline $z_{t+1} \approx z_t$ is a strong solution that doesn't need actions at all. If that's true at the data level, action-conditioned prediction degenerates: $T_\omega$ learns to ignore $a_t$ and copy $z_t$ forward, achieving a low loss without ever using action information. Position-decode wouldn't necessarily improve from this kind of training.

The fix is to require, at the dataset level, that the chosen training environment have $r_\text{action} > 0.15$ — meaning conditioning on action reduces one-step MSE by at least 15% relative to history-only prediction. This is a *preflight* check on the env, run before any substrate training. If $r_\text{action}$ is below the threshold, change the env or change the action sampling distribution (more exploration, less smooth trajectories) until it's above. **Evidence-driven correction from the four-fix arc.**

This is a methodology fix more than an architectural one, but it's the single most important thing to get right. Most substrate failures on smooth manipulation video come from action_leverage being too low to make the action-conditioned loss informative.

### Auxiliary 2: Position-decode head as diagnostic only (evidence-driven correction)

A small position-decode head is trained jointly with the substrate, but with a *detached* gradient — the head's loss does not flow back into $f_\theta$ or $T_\omega$. The head is purely a diagnostic that surfaces, during training, whether the state is becoming position-decodable.

$$\mathcal{L}_\text{decode-aux}(\eta) = \mathbb{E}_{x_t}\left[\|D_\eta(\text{sg}[f_\theta(x_t)]) - p_t\|_2^2\right]$$

where $p_t$ is ground-truth agent position (available in simulated envs), $D_\eta$ is the decoder head, and the stop-gradient prevents the decode loss from training the substrate.

The role: this is a *training-time Gate 0 estimator*. If after 5k training steps the auxiliary decode is still at chance level, the substrate isn't learning position even with action conditioning, and we should stop and reconsider rather than burn another 95k steps. This is the cheapest possible early-stopping diagnostic. **Evidence-driven correction.** The detached gradient is critical — without it we'd be teaching to the test.

### Auxiliary 3: Temporal-contrastive at multiple scales — CUT (v2)

**CUT in v2.** The original motivation was that single-scale prediction might miss
non-action-conditioned motion components, and multi-scale contrast at $k\in\{1,4,16\}$
would catch them. Korchinski, Favero & Wyart (2026), *"Learn from your own latents
and not from tokens: A sample-complexity theory"* (arXiv:2605.27734), remove that
motivation directly: on data with hierarchical latent structure, a **single-level**
latent-prediction objective (data2vec-style, EMA target) **already performs implicit
level-by-level latent clustering** and recovers the full hierarchy at sample
complexity $\sim m^3$ — *constant in tree depth $L$* — vs $m^{L+1}$ for token-level
SSL. So a single-scale objective is already implicitly multi-scale on hierarchically-
structured data; the explicit multi-scale term is redundant work, not insurance.
Reacher dynamics (joint angle → pose → trajectory → task) are hierarchically
structured, so the qualitative prediction applies (with the caveat in §9 that the
exact $m^3$ bound is proven only on RHM data, which Reacher only approximately
satisfies). This is a *cut*, not a "speculative, ablate later" — the theory and its
data2vec evidence are direct. The ablation matrix (§8) loses its `+temp` row.

### Auxiliary 4: Per-dimension VICReg variance hinge (evidence-driven correction)

From the experimental arc, the LeJEPA-default SIGReg at $\lambda = 0.1$ was insufficient at small data scales — the per-dimension VICReg variance hinge has an attractor at "real variance" that pure moment-matching doesn't. Use the hinge:

$$\mathcal{L}_\text{var}(\theta) = \frac{1}{d_z}\sum_{d=1}^{d_z} \max(0, 1 - \sqrt{\text{Var}(z_d)})$$

Per-dimension floor on standard deviation. Combine with SIGReg if you want both moment-matching and the floor; use just the hinge if you want to keep the loss list short.

**Evidence-driven correction from the experimental arc.**

## 4. Architecture

Three components, kept deliberately simple because the contribution is the objective, not the architecture:

**Encoder $f_\theta$.** ViT-S or ViT-B (22M or 86M params). No pretrained weights. Patch size 8 or 16 depending on input resolution. Standard ViT, no architectural cleverness. The point is to test whether the objective produces a motion-tracking substrate, not whether a clever encoder makes the objective work better. If a clever encoder is needed, that's a finding, not a feature.

**Latent dynamics $T_\omega$.** 4-layer transformer that takes $(z_t, a_t)$ and produces $\hat z_{t+1}$. Action is embedded by a small MLP $E_a$ first. Same dimensionality as $z$. Small enough to be inference-cheap, large enough to model nontrivial dynamics.

**EMA target encoder $\bar f$.** Standard self-distillation pattern. Momentum 0.998 → 0.9999 annealed.

**Position-decode head $D_\eta$.** 2-layer MLP, 256 hidden units. Trained on $\text{sg}[f_\theta(x_t)]$ to predict $p_t$. Diagnostic only.

Total parameter count: roughly 30-100M depending on encoder size. Cheap to train. Runs on 1-2 H100s in 12-48 hours depending on env and dataset size.

## 5. Environment selection

The four-fix arc happened on OF-JEPA-navigate. The substrate rewrite should be on a different env, both because the existing env's data may be insufficient for the new objective and because changing one variable at a time is good practice.

Three candidates ranked by suitability:

**Top choice: DMControl Reacher / Cheetah / Walker.** Action-leverage is high (every action visibly moves the agent), dynamics are nontrivial (contacts, joint constraints), position is well-defined (joint angles map cleanly to spatial position). Fast simulation, cheap data collection, well-established benchmark. The Reacher task specifically gives you a 2D position target that's directly comparable to the Gate 0 threshold.

**Second choice: ManiSkill Pick-and-Place.** Harder env, more realistic dynamics, but action_leverage is medium-low and the position-decode target is less clean (which point on the object counts as position?). Worth using after DMControl confirms the objective works.

**Third choice: A custom moving-agent env with controllable smoothness.** If you want to specifically test the action_leverage hypothesis from the experimental arc, build a parametric env where you can dial the smoothness up and down and measure the Gate 0 outcome at each setting. This is the most informative experiment but the slowest to set up.

Start with DMControl Reacher. Pass Gate 0 there. Then expand.

## 6. The Gate-0 precommit harness

The harness is a standalone module that takes a frozen substrate checkpoint and produces a pass/fail. Spec:

**Inputs.**
- Path to substrate checkpoint
- Path to held-out evaluation dataset (frames + ground-truth positions)
- Decoder config (default: 2-layer MLP, 256 hidden, trained for 5k steps)

**Procedure.**
1. Load checkpoint, freeze encoder $f_\theta$.
2. Embed every held-out frame: $z_i = f_\theta(x_i)$.
3. Split held-out set 80/20 into decoder-train and decoder-test.
4. Train fresh decoder $D_\eta$ on the 80% subset, predicting $p_i$ from $z_i$. 5k steps, fixed hyperparameters.
5. Evaluate decoder on 20% test subset. Compute mean L2 error in pixels.
6. Pass if mean error < 5px. Fail otherwise.

**Output.**
- Pass/fail boolean
- Mean L2 error in pixels
- Per-axis error (x, y separately) for diagnostic
- A scatter plot of predicted vs. true position, saved as artifact
- A JSON record with checkpoint hash, dataset hash, error metrics, timestamp

**Critical design choices.**

The decoder is *fresh* — re-initialized for every evaluation. No carryover from a previous decoder. This prevents the gate from being passed by progressively fitting a decoder that doesn't reflect the substrate.

The decoder training is *deterministic given the same seed* — fixed steps, fixed batch order, fixed initialization. Two runs of the harness on the same checkpoint should produce the same answer to within numerical noise.

The harness is *cheap*. Should complete in <10 minutes on a single GPU. This is critical because it gets run constantly during substrate training as an early-stopping diagnostic. If it takes hours, it stops getting run.

The harness writes its output to a *standardized JSON schema* that can be aggregated across experiments. This is the precommit record — every substrate checkpoint that gets built upon has a Gate 0 record committed alongside it. No substrate enters downstream training without a passing record.

**File location.** `gates/gate0_precommit.py` per your earlier naming. Standalone, importable. Should also have a CLI entry point for one-off evaluation: `python -m gates.gate0_precommit --checkpoint X --data Y`.

## 7. Training script

Single file: `system1_motion/train.py`. Roughly 300 lines including the data loader, the loss assembly, the training loop, and the periodic Gate 0 diagnostic.

The loss assembly:

```python
def compute_loss(batch, model, hyperparams):
    z_t       = model.encoder(batch.x_t)
    z_tp1_tgt = stop_grad(model.target_encoder(batch.x_tp1))
    
    # Primary: action-conditioned prediction
    z_tp1_pred = model.dynamics(z_t, batch.a_t)
    L_pred = normalized_mse(z_tp1_pred, z_tp1_tgt, model.running_sigma_z)
    
    # Auxiliary 1: action-leverage diagnostic (logged only, not in loss)
    log_action_leverage(model, batch)
    
    # Auxiliary 2: position-decode diagnostic (detached)
    p_pred = model.decoder(stop_grad(z_t))
    L_decode_aux = mse(p_pred, batch.p_t)
    
    # Auxiliary 3: temporal contrast (optional)
    if hyperparams.use_temp_contrast:
        L_temp = multi_scale_contrast(model.encoder, batch, scales=[1, 4, 16])
    else:
        L_temp = 0
    
    # Auxiliary 4: VICReg variance hinge
    L_var = variance_hinge(z_t)
    
    # Total. Decode head trains independently.
    L_substrate = L_pred + hyperparams.alpha * L_temp + hyperparams.beta * L_var
    L_decoder = L_decode_aux
    
    return L_substrate, L_decoder
```

The substrate loss and decoder loss are computed jointly but back-propagated through *separate optimizer steps*. The substrate optimizer sees only $L_\text{substrate}$. The decoder optimizer sees only $L_\text{decoder}$. Stop-gradient on $z_t$ in the decoder path enforces this at the autograd level as well.

**Training schedule:**
- Steps 0-2000: warmup. EMA momentum ramps from 0.99 to 0.998. Variance hinge weight $\beta$ ramps from 0 to 1.0. Temporal contrast weight $\alpha = 0$.
- Steps 2000-10000: base training. Action-conditioned prediction is the dominant signal.
- Step 5000: first Gate 0 precommit check via the harness. **Hard stop if not improving over chance** (chance level on DMControl Reacher is roughly 30-50px mean error; failure to improve below 25px by step 5k is the kill signal).
- Steps 10000-30000: continued training. Add temporal contrast at $\alpha = 0.3$ if step-5000 Gate 0 showed progress but not pass.
- Step 15000, 25000: Gate 0 precommit checks.
- Step 30000: final Gate 0 check. Pass/fail.

**Total compute budget: 30k steps on DMControl Reacher.** At ViT-S scale, this is roughly 8-12 hours on a single H100. Cheap enough that running it 3-5 times with different seeds to confirm robustness is feasible.

## 8. Ablation plan

After cutting Aux 3 (v2), the matrix is three auxiliaries; each must earn its place:

| Run | Primary | Aux 1 (env preflight) | Aux 2 (decode diag) | Aux 4 (var hinge) |
|---|---|---|---|---|
| A1 (base) | ✓ | ✓ (env passes leverage) | ✓ (diagnostic only) | ✗ |
| A3 (+hinge) | ✓ | ✓ | ✓ | ✓ |
| B1 (low leverage) | ✓ | ✗ (env fails leverage) | ✓ | ✗ |
| B2 (no decode diag) | ✓ | ✓ | ✗ | ✓ |

A1 is the minimum viable run. If A1 passes Gate 0, `+hinge` (A3) needs to show
*additional* improvement to be kept. (The `+temp` rows are removed — Aux 3 cut, §3.)

B1 is the negative control — running on an env that fails the leverage preflight to verify that the preflight is actually predictive of Gate 0 failure. This validates the methodology lesson from the experimental arc.

B2 ablates the decode diagnostic — does training-time visibility into Gate 0 actually help the experimenter steer toward a passing substrate, vs. running blind and checking at the end? Probably yes, but worth measuring.

The ablation matrix is what turns "we built a substrate" into "we identified which components of the substrate matter." That's the paper-shaped output.

## 9. What this spec is and isn't

**What it is:**
- A minimum viable substrate that the four-fix-arc evidence says should pass Gate 0
- A methodology (env preflight + training-time decode diagnostic + standalone precommit harness) that prevents the failure mode the arc demonstrated
- An ablation matrix that produces publishable findings regardless of which auxiliaries earn their place
- **A single-level objective by design — and that choice is now theory-backed, not just simpler.** Korchinski, Favero & Wyart (2026, arXiv:2605.27734) prove that a single-level data2vec-style latent-prediction objective with an EMA target implicitly performs hierarchical, level-by-level latent clustering, recovering the full latent tree at sample complexity $\sim m^3$ — *constant in depth* — vs $m^{L+1}$ for token-level SSL. We chose single-level not for simplicity but because the theory says a well-targeted single-level objective is *already implicitly hierarchical* on hierarchically-structured data. (Caveat: proven on RHM data; see §11.)

**What it isn't:**
- A claim to be "best in class." It's a claim to be the right next move given the evidence. Best in class is a comparison the field hasn't done and that requires baseline runs we haven't scoped.
- A complete BLA roadmap. This is the substrate rewrite. The bicameral roadmap (System 2, router, RAM, etc.) sits downstream of this passing.
- A novel objective. Action-conditioned latent prediction is V-JEPA 2-AC. The novelty is in the methodology around it (env preflight, training-time decode diagnostic as detached signal, precommit harness as a hard gate), not in the loss function.
- **A claim about *hierarchical representation learning* — and this must not be conflated with BLA's bicameral *planning* claim.** Korchinski et al. weaken the case for *naive stacked H-JEPA* (one encoder per scale, independent losses); they explicitly put multi-scale-teacher variants (V-JEPA 2.1, Bootleg) out of scope. BLA's System 2 is a **latent-diffusion bidirectional planner**, not a coarser-scale JEPA — the redundancy result does not reach it, nor the router/RAM/prefetcher. The correct reading: keep System 1 a **single-scale** substrate (the paper says stacking it buys little), and justify the bicameral *planning* split separately. Any future BLA writeup must keep "single-level representation is sufficient (per Korchinski)" distinct from "bicameral planning is valuable (justified on planning semantics)."

**Optional post-hoc diagnostic (stronger than Gate 0 alone).** Korchinski et al.'s Figure 5 gives a synonym-clustering probe that tests whether a trained substrate is doing the *level-by-level latent clustering* the theory predicts. Our EMA target encoder is exactly the data2vec mechanism their analysis covers, so the probe applies. Gate 0 tells you position is recoverable; the clustering probe tells you *whether the implicit-hierarchy mechanism is operating* — running both gives a much stronger story. Worth adding after A1 passes Gate 0; not a blocker.

The discipline I'm trying to enforce is: every design choice traces to either standard recipe, evidence from the four-fix arc, published theory, or a flagged speculative test. No design choice that's there because it sounds good. The arc cost real GPU time to surface the lesson about objectives-vs-regularizers; this spec is what honoring that lesson looks like.

## 10. Order of operations

1. Pause the OF-JEPA-navigate pod. Save artifact directory. Tear down. **Today.**

2. Pick env. Run the action-leverage preflight standalone, before any substrate training. Confirm $r_\text{action} > 0.15$ on chosen env. **Day 1-2.**

3. Implement Gate 0 precommit harness. Test it against a known-failing substrate (the existing OF-JEPA-navigate checkpoint) to confirm it produces "fail." Test against a known-passing reference if one exists; if not, accept that the first passing substrate is also the first validation of the harness. **Day 2-3.**

4. Implement training script. Run A1 (base) for 30k steps. **Day 3-5.**

5. Gate 0 evaluation on A1 final checkpoint. If pass: proceed to ablations. If fail: diagnose using the training-time decode trace and decide whether to add auxiliaries or rethink. **Day 5-6.**

6. Ablation matrix runs. Each is 8-12 hours on one GPU. Parallel if hardware allows. **Day 6-10.**

7. Writeup. Either a positive result (substrate passes, here's the recipe) or an honest-failure result (substrate doesn't pass at this scale, here's what we learned about why and what would need to change). **Day 10-14.**

Two weeks to a result, positive or negative. If the result is positive, the bicameral roadmap regains ground to stand on. If the result is negative, the next iteration is informed by a sharper failure mode than "the substrate is collapsed."

## 11. The honest assessment

I think A1 will pass Gate 0 on DMControl Reacher. Action-conditioned prediction is a well-established objective, Reacher has the leverage to make it informative ($r_\text{action}=0.996$, measured), and Korchinski et al. (2026) give theory that a single-level latent-prediction objective is implicitly hierarchical on hierarchically-structured data. Probability estimate, bumped from 70% to **75–80%** on that theoretical backing.

**The caveat that keeps it at 75–80% and not higher:** the $m^3$ guarantee is proven only on **RHM data** (fixed tree topology, non-recursive, unambiguous grammar). Reacher dynamics are continuous, contact-aware, and only *approximately* hierarchical — the exact bound does not literally apply. What transfers is the *mechanism* (EMA-target single-level prediction → implicit level-by-level clustering), and there is **no experimental evidence in that paper that the result transfers off RHM** (their data2vec experiments are all RHM-distributed). So this is a stronger prior, not a guarantee.

My prediction:
- A1 passes Gate 0 at roughly 3–4px on Reacher (~60%)
- A1 passes but only barely, 4–5px (~20%)
- A1 fails (~20%) — and because Aux 3 is now **cut**, the failure branch is *not* "add temporal contrast." It becomes the genuinely interesting question: **does the RHM implicit-hierarchy result fail to transfer to non-RHM (robotics) data, and what additional machinery does real-world data require?** That negative result is publishable (Option B framing): "the single-level theory holds on RHM but needs X on continuous contact dynamics."

Either branch is publishable. Positive: a substrate-methodology result whose single-level choice is theory-backed. Negative: the first empirical test of whether the Korchinski et al. implicit-hierarchy mechanism transfers off RHM — a question that paper explicitly leaves open.

Either branch produces a publishable result. The positive branch is a methodology paper: "Gate-driven substrate development for embodied JEPA, with preflight and decode-diagnostic harness." The negative branch is "Action-conditioning alone fails on smooth manipulation video; here's the ablation showing what works." Both are useful to the field. Both are honest.

What this spec does *not* promise: that BLA will beat LeWM on Push-T or VLAs on CALVIN. Those are downstream claims that require runs we haven't designed. This spec is the prerequisite the bicameral roadmap was missing. Pass the gate, then the rest of the roadmap has ground to stand on. Don't pass the gate, and the roadmap was premature regardless of the architectural ideas it contained.

That's the substrate spec. Methodologically tight, evidence-driven, with a few speculative additions clearly marked. Two weeks to a result. No "best in class" claim until baseline runs justify it.
