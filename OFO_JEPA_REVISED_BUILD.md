# Operator-JEPA — revised build spec (the "keep a third" design)

Thesis: **clean prediction-substrate + action-grounded operator + a return-selector.** Keep the
operator, the factorization, belief-state U, counterfactual/Lean triples, and the gated ladder.
Cut the typed-token zoo and the precision/cancel-the-predictable losses. Everything below is marked
✓ keep / △ changed-from-prior / ✗ cut, with the reason.

---

## Keystone — predict the OPERATOR, not the next state  ✓
`Ω_t = D(z_struct_t)` ; `z_struct_{t+1} = Ω_t(T(a_t), z_struct_t)`, `T` = learned action→operator-coord map
(corollary-discharge transform). Convergence of three threads: predict-the-transformation, TEM
action-indexed weights, cerebellar forward model — same object.

Why it fixes rollout: naive rollout averages over actions, so controllable state sits in irreducible
conditional variance and gets no gradient. Conditioning on `a_t` collapses that variance and FORCES
the encoder to carry controllable state. We did not falsify "predict the future"; we falsified
predicting it with the action *outside* the generator.

△ **Corrected justification (don't overstate it).** The identifiability theorem needs TWO premises:
(a) stationary additive-noise transitions, (b) isotropic-Gaussian latents. Action-conditioning
restores **(a)** only — given `a_t` the operator residual is stationary. The build deliberately
forfeits **(b)** for z_struct. So **z_struct stays OUTSIDE the theorem** — grounded by action-conditioning
alone, no identifiability guarantee. The guarantee lands on z_content (the visible, already-groundable
half). Consequence: z_struct's only guarantee is the probe → **Gate-0 is load-bearing, not decorative.**

---

## Architecture — factored z = (z_content, z_struct)  ✓ with △

### z_content  ✓
Slots (SCOF), bound by **synchronization (AKOrN-style)** so binding survives masked views (the open
obstacle). Regularize to clean isotropic Gaussian (SIGReg); leave it clean — this is the identifiable,
reusable content code.
△ **Tension to log, not hide:** slots are non-isotropic by construction, so the theorem certifies
*within-slot* content at best; cross-slot binding sits *outside* the guarantee. Don't claim more.

### z_struct  ✓ with △
Low-dim; never free-regressed; only ever the state an action-indexed operator acts on. All structural
and predictive-information constraints live on **Ω and the decode heads — never on the base distribution.**
△ **The factorization needs a sponsor, and it's the bottleneck.** What forces content→z_content,
struct→z_struct is *only* the z_struct width + operator grounding. PRE-REGISTER `dim(z_struct)` **tight,
below the controllable-state dimension** (the d_b=4 forced-choice lesson). Above threshold, content
leaks in and the split decorates. Width is a registered claim, not a free knob.

---

## Grounding sponsors — both put action in the loop  ✓
- forward operator prediction (the conditional-variance collapse), and
- inverse-dynamics readout off z_struct.
The "slots only partially revive inverse dynamics" was the low action-dim; the forward operator + T
is what strengthens it.

---

## CUT  ✗
**Precision-weighting / cancel-the-predictable.** Reason: precision = inverse-variance → it amplifies
low-conditional-variance dims (controllable — already done by the operator, redundant) and **dampens
high-variance dims = exactly the unpredictable decision-relevant residue.** It cannot tell high-variance
nuisance from high-variance signal; it suppresses both. It is structurally incapable of helping — and
capable of burying — the one variable class that's actually hard (the residue), which only RETURN reaches.
Don't reframe it; cut it.

**The five typed tokenizers / declared Z = {F,O,R,G,U,Ω}.** Reason: naming a lane doesn't ground it —
our rollout/decode result in a new costume. A typed lane carries what you want only if something
sponsors it (action-in-loop or decode). Four of six are unsponsored declarations that collapse into the
aggregate loss or quietly need their own decode sponsor. TEM earns what/where because where is
action-driven — structure as a *consequence of dynamics*, not a declaration. Add lanes one at a time,
each earning its slot by passing its gate (ladder below).

---

## Belief-state U  ✓ with caveats
U is a real **predictor latent variable** (energy/latent-variable predictor, NOT a hard top-k set loss,
which mode-drops). Train across context budgets; supervise uncertainty so partial obs changes
*confidence*, not collapses to a point estimate.
△ Caveats: calibrating against "true ambiguity" needs the true posterior — available in **sim/Lean,
not in general**; and the ambiguity that matters is **reward-relevant** ambiguity, so even U leans on return.

---

## Counterfactual / Lean operator — BUILD THIS FIRST  ✓ (the hard-axis core)
`(proof_state, tactic, verified_next_state, valid?)` triples ground the operator far harder per sample
than passive video, **because the verifier IS the return-selector.** Operator = tactic-transition
dynamics (the proposal); Lean/z3 verifier = truth (the selector); soundness gate rejects
unsound-but-helpful operators. This is the LLM-proposes / verifier-decides principle made into the
training signal — and the one place the return is already in the loop. Reuses the existing proofworld
trust boundary.

---

## The falsifiable bet
Minimal `(z_content, z_struct)` + a single grounded operator **matches or beats the full six-type stack**
on Gate-0 controllable-state **usability** and on contact-sensitive planning, at a fraction of the
machinery. If the full typed stack wins on controllable usability, the typing carries real weight and
the bet is wrong — clean experiment, settles it.

---

## Ladder — most-decisive-first, every rung gated on USABILITY not decode  △
**Gate 0 (prerequisite, before anything promotes):** controllable-state **usability** on z_struct via the
matched-capacity closed-loop arbiter (~/grounding_meter/usability_probe.py). △ NOT decode — decode lies =
capacity mismatch; a non-isotropic action-grounded z_struct can pass decode and fail usability.

- **A.** Dense/deep base prediction, clean SIGReg on z_content only (V-JEPA-2.1 all-token dense loss — keep).
- **B.** Action-indexed operator + inverse-dynamics readout. Gate: action-effect + counterfactual accuracy; `T(a)` beats raw action-concat (falsifiable — if concat ties, drop T).
- **C.** Belief-state U with calibrated uncertainty across budgets. Gate: calibration under partial obs, no mean-collapse on multimodal futures. (was "precision-weighting" — replaced; precision cut.)
- **D.** Add exactly ONE typed lane — events/contacts first (where planning breaks). Gate: must decode AND be usable for a held-out contact variable to earn its place. Only then consider relations.

Each typed token earns its slot by grounding+usability, or it isn't built.

---

## Through-line
Right because: clean prediction-substrate (z_content + operator z_struct) + return-selector (Lean
verifier, counterfactuals). Every remaining soft joint is the same fault — where there's no return the
substrate can't reach the residue: precision-weighting buries it, U's calibration needs the posterior,
the theorem certifies only the visible half. So **build the verifier-grounded proof operator first**
(return in the loop, trust boundary owned); the physical `(z_content, z_struct)` build is the
falsifiable bet — run it, gate on **usability**, and don't expect substrate tricks to ground
decision-relevance, because nothing self-supervised does.
