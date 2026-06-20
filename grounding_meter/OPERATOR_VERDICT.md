# Operator-JEPA keystone — built & tested (synthetic, 2026-06-19)

Tested "predict the OPERATOR, not next state" in controlled testbeds (operator_test.py, operator_test2.py).
Three models compared at matched capacity: rollout (no action in generator), concat (action stapled),
operator (Ω=hypernet(z) emits affordance W(z), applied to T(action)).

## Results
1. VISIBLE controllable state: rollout == concat == operator (c-decode gap ~0 at every bottleneck k).
   Action-in-generator adds NOTHING — rollout already grounds visible+persistent controllable state
   (reproduces EFFERENCE_NEGATIVE_RESULT: vanilla grounds the arm, SNR not sponsor).
2. INVISIBLE config-determining state (steelman, efference regime): concat/operator GROUND it (~0.44)
   where rollout FAILS (~0.15) — gap +0.30. The keystone is real but NARROW: action-in-generator
   matters ONLY when the controllable state isn't already recoverable from the prediction target.
3. OPERATOR vs CONCAT: ~equal on grounding (0.40 vs 0.44) AND on transfer to held-out configs
   (0.04 vs 0.03, both fail). The operator parameterization = PURE RE-PARAMETERIZATION here; no
   grounding bias, no generalization bias.
4. RESIDUE (uncontrolled + invisible u): ungrounded by all (0.00) — needs return.

## Verdict for the build
- KEEP "action inside the generator" (concat suffices) — grounds controllable in the regime that matters.
- DEMOTE "predict the operator": the OPERATOR factorization (the novel architectural claim) shows no
  measured advantage over concat. It must EARN its complexity with a transfer/usability win — which this
  test does NOT show. Default to concat (Occam) until a richer-affordance world (contact, many objects)
  demonstrates the operator's inductive bias. The burden is on the operator.
- Residue still return-bound (confirmed).
CAVEAT: synthetic 1D-gain config is simple; the operator's bias might appear in richer affordance
structure — but that's the test that must be passed, not assumed.

## CORRECTION (3-seed, the single-seed call was too strong)
Operator vs concat on richer bilinear-affordance world, held-out transfer, 3 seeds:
- concat   held-out pred-MSE 1.94±0.05 | afford-decode 0.51±0.01
- operator held-out pred-MSE 1.62±0.31 | afford-decode 0.52±0.01  (operator-minus-concat per seed: -0.07,-0.67,-0.22, ALL negative)
REFINED VERDICT: the operator's inductive bias is REAL but MODEST and PREDICTOR-SIDE ONLY (~16% lower
held-out MSE, consistent sign, high variance). NOT a grounding win (affordance-RECOVERY ties — the
encoder infers tau equally). Neither solves compositional transfer (both 6x blowup). The bottleneck is
the SHARED ENCODER's affordance-inference extrapolation, which operator-on-the-predictor can't move.
=> Keystone grounding stays "action-in-generator" (concat gets it). Operator = a modest predictor bias,
keep if cheap, NOT the architecture's load-bearing piece, NOT a transfer solution. The real open problem
is encoder affordance-inference extrapolation. (My single-seed "tie=>drop it" was the seed-discipline error.)
