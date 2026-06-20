# Grounding phase-diagram meter — results (single-seed rig unless noted)

Constructed decision-relevant variable z; 3 dialable axes (alpha=prediction-variance,
controllability, pivotality); 3 grounding arms (prediction, efference, return).

## Four-band encoding phase diagram (decode-R2 of z per arm) — 4/4 pre-registered hits
- Band 1 (prediction): z visible (alpha>~0.45) -> predZ 0.84->0.98. knee same for ctrl/exo.
- Band 2 (efference): controllable z, alpha in [0.15,0.45] -> effZ 0.48->0.91 while predZ=0.
  Exogenous z: effZ<=0.03 (no purchase). [ground2.py]
- Band 3 (return): invisible+uncontrolled z, DENSE reward -> ~0.45 (low+lossy ceiling, sqrt/sign).
- Band 4 (residue): invisible+uncontrolled+SPARSE -> ~0.12 ~ ungrounded. [ground5.py]
- Cost in band 3 driven by PIVOTALITY (consequence frequency), not sample count.

## Band-1 USABILITY (closed-loop, matched-capacity) — the falsifier, PASSED
- clean band-1 usability = 0.88 +/- 0.01 (lin) / 0.89 +/- 0.00 (mlp), 4 seeds. >= 0.8 pass. [seedrep.py]
- U3: usability tracks decode AT CONTROLLER CAPACITY across tangle sweep -> "decode lies" = capacity mismatch.
- closed-loop usability can EXCEED static decode (temporal integration) -> decode is two-sided-biased proxy.
- U2 limitation: static scalar can't make MLP-decode>>lin-decode; subtle phase-memory NEEDS real-rig temporal/high-dim structure.

## Status
Cheap-substrate thesis (prediction-substrate + return-selector; residue = SSL-blind AND return-silent corner)
SURVIVES its falsifier in-rig. Still a METER, not a universal result: port to real rig,
run a real controllable + a real prediction-invisible variable with usability overlay live.
TODO: seed-rep band-3 magnitudes; selector-capacity axis; real-rig port.
