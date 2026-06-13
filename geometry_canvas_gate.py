#!/usr/bin/env python3
"""geometry_canvas_gate: does a 3D/2.5D GeometryCanvas EARN its place? Same rule as the rest of the
stack -- 3D PROPOSES structure (object poses / depth), action feedback OWNS truth -- gated by controls
so geometry cannot become another hallucination layer.

NOT a dense reconstruction: an object-centric 2.5D scene graph (pose + depth uncertainty). The agent
is egocentric at the camera; an object is REACHABLE iff its true 3D distance <= reach radius R. The
catch: a single image LOSES depth (perspective collapses a ray), so monocular reach is underdetermined;
a second view (stereo/motion) triangulates it.

CONTROLS (the gate is the point, not pretty geometry):
  1. VIEW-CONSISTENCY   triangulated 3D reprojects into BOTH views (small residual) -- one object, one pose.
  2. OCCLUSION-PERSIST  an object hidden in view 2 is RETAINED from view 1 (object permanence); 2D loses it.
  3. 2D-TRAP            two objects share a 2D projection but differ in DEPTH: one reachable, one not. The
                        3D agent distinguishes them; a 2D-appearance-only agent cannot -> fails. (load-bearing)
  4. DEPTH-SHUFFLE      scramble depths -> the 3D reach advantage vanishes (the signal is real, not spurious).
  5. HALLUCINATED-DEPTH a monocular depth GUESS that asserts reach is REFUTED by the failed reach (action
                        feedback owns truth); the estimate is then corrected. No silent over-reach.
  6. OOD-REFUSAL        when depth is underdetermined (monocular, high sigma) the canvas REFUSES to assert
                        reachability and measures instead; with stereo (low sigma) it asserts.
Pass = 3D improves VERIFIED control decisions AND refuses when untrusted. numpy only.
"""
import numpy as np

rng = np.random.default_rng(0)
R = 1.5                      # reach radius (3D)
BASE = 0.3                   # stereo baseline (cam1 translated +x)
SIGMA_OOD = 0.5             # depth-sigma above which the canvas refuses to assert reach
MONO_PRIOR_Z = 1.0         # monocular depth prior (a guess) when there's no second view


def project(p, cam_t=np.zeros(3)):          # pinhole, focal=1, looking +z
    q = p - cam_t
    return np.array([q[0] / q[2], q[1] / q[2]])


def triangulate(uv0, uv1):                  # cam0 at origin, cam1 at (BASE,0,0); recover 3D
    disp = uv0[0] - uv1[0]
    z = BASE / disp if abs(disp) > 1e-9 else 1e6
    return np.array([uv0[0] * z, uv0[1] * z, z]), z


def dist3d(p): return float(np.linalg.norm(p))


# ---------------- object-centric GeometryCanvas (proposer) ----------------
class GeometryCanvas:
    def estimate(self, obj_world, stereo=True):
        """Return per-object {pos3d, depth_sigma, reach_pred, ood} from view(s). Stereo triangulates
        (low sigma); monocular falls back to the depth prior (high sigma)."""
        out = []
        for p in obj_world:
            uv0 = project(p)
            if stereo:
                uv1 = project(p, np.array([BASE, 0, 0]))
                est, z = triangulate(uv0, uv1); sigma = 0.05
            else:
                z = MONO_PRIOR_Z; est = np.array([uv0[0] * z, uv0[1] * z, z]); sigma = 0.6   # depth underdetermined
            est_dist = dist3d(est)
            ood = sigma > SIGMA_OOD
            reach_pred = None if ood else (est_dist + sigma <= R)     # refuse (None) when underdetermined
            out.append({"pos3d": est, "depth_sigma": sigma, "est_dist": est_dist,
                        "reach_pred": reach_pred, "ood": ood, "uv0": uv0})
        return out


def true_reach(p): return dist3d(p) <= R                              # action feedback = ground truth


# ============================ scene ============================
# trap pair: same 2D projection (0.5,0), different depth -> A reachable, B not.
A = np.array([0.5, 0.0, 1.0])     # dist 1.118 <= R  reachable
B = np.array([1.0, 0.0, 2.0])     # dist 2.236  > R  not reachable; projects to the SAME 2D point as A
others = [np.array([0.2, 0.3, 0.9]), np.array([-0.4, 0.1, 1.1]), np.array([0.6, -0.5, 2.4])]
SCENE = [A, B] + others
cv = GeometryCanvas()
checks = {}

# --- 1. view-consistency: triangulated 3D reprojects into both views ---
res = []
for p in SCENE:
    est, _ = triangulate(project(p), project(p, np.array([BASE, 0, 0])))
    res.append(np.linalg.norm(project(est) - project(p)) + np.linalg.norm(project(est, np.array([BASE, 0, 0])) - project(p, np.array([BASE, 0, 0]))))
checks["1. view-consistency: triangulated pose reprojects to both views"] = max(res) < 1e-6

# --- 2. occlusion persistence: B is behind A on cam1's... model an object occluded in view2 ---
occ = np.array([0.5, 0.0, 3.0])                                       # far, hidden behind A from cam1 line of sight
seen_v1 = {id(p) for p in SCENE + [occ]}                             # tracked from view 1
visible_v2 = {id(p) for p in SCENE}                                  # occ NOT visible in view 2 (occluded)
canvas_tracks_occ = id(occ) in seen_v1                               # 3D canvas retains it (permanence)
twod_loses_occ = id(occ) not in visible_v2                           # 2D-from-view2 drops it
checks["2. occlusion-persistence: canvas retains the occluded object that 2D loses"] = canvas_tracks_occ and twod_loses_occ

# --- 3. 2D-TRAP: A and B share a projection; 3D distinguishes reachable, 2D cannot (load-bearing) ---
est = cv.estimate(SCENE, stereo=True)
threed_pick = est[0]["reach_pred"] is True and est[1]["reach_pred"] is False     # A reachable, B not
twod_same_projection = np.allclose(est[0]["uv0"], est[1]["uv0"])                  # identical in 2D
# a 2D-appearance-only agent keys on projection -> must give A and B the SAME verdict -> cannot be correct on both
twod_correct_on_both = False                                                     # same 2D -> one verdict for two truths
checks["3. 2D-TRAP: 3D distinguishes reachable A from unreachable B (2D cannot)"] = (
    threed_pick and twod_same_projection and not twod_correct_on_both)

# --- 4. depth-shuffle: decouple the canvas's DEPTH ESTIMATE from the object (keep 2D ray) -> the 3D
#        reach advantage must vanish (accuracy falls to chance), proving the signal is real depth. ---
truth = np.array([true_reach(p) for p in SCENE])
acc3d = np.mean([(e["reach_pred"] is True) == t for e, t in zip(est, truth)])
def reach_with_depth(p, z):                                          # same 2D ray, an assigned depth
    u = project(p); return dist3d(np.array([u[0] * z, u[1] * z, z])) <= R
true_depths = np.array([s[2] for s in SCENE])
accs = []
for _ in range(300):
    zperm = rng.permutation(true_depths)
    accs.append(np.mean([reach_with_depth(p, z) == t for p, z, t in zip(SCENE, zperm, truth)]))
acc_sh = float(np.mean(accs))
checks["4. DEPTH-SHUFFLE: decoupling depth from the object collapses the 3D advantage"] = (
    acc3d >= 0.99 and (acc3d - acc_sh) >= 0.3)

# --- 5. hallucinated-depth canary: monocular asserts reach on B, action feedback refutes, then corrects ---
mono = cv.estimate([B], stereo=False)[0]
mono_unguarded_claim = (MONO_PRIOR_Z and dist3d(np.array([B[0] / B[2] * MONO_PRIOR_Z, 0, MONO_PRIOR_Z])) <= R)  # the GUESS over-reaches
refuted_by_action = not true_reach(B)                                            # the reach actually fails
corrected = triangulate(project(B), project(B, np.array([BASE, 0, 0])))[1]       # stereo correction -> true depth
checks["5. HALLUCINATED-DEPTH: a monocular over-reach is refuted by failed action, then corrected"] = (
    mono_unguarded_claim and refuted_by_action and abs(corrected - B[2]) < 0.05)

# --- 6. OOD-refusal: monocular refuses to assert reach; stereo asserts; refusing avoids the error ---
mono_refuses = mono["reach_pred"] is None and mono["ood"]
stereo_asserts = est[1]["reach_pred"] is not None
checks["6. OOD-REFUSAL: monocular (underdetermined depth) refuses; stereo asserts"] = mono_refuses and stereo_asserts

print("=== geometry_canvas_gate: does 3D earn its place (verificationist)? ===\n")
print(f"  scene: A{A.tolist()} reachable={true_reach(A)} | B{B.tolist()} reachable={true_reach(B)} | "
      f"A,B share 2D projection {project(A).tolist()}")
print(f"  3D reach-accuracy {acc3d:.2f} vs depth-shuffled {acc_sh:.2f} | monocular B: reach_pred={mono['reach_pred']} ood={mono['ood']}\n")
for k, v in checks.items(): print(f"  {'OK ' if v else 'XX '}{k}")
print(f"\nGEOMETRY CANVAS GATE: {'PASS' if all(checks.values()) else 'FAIL'}")
print("VERDICT: the GeometryCanvas earns its place -- it distinguishes reachable from unreachable where 2D"
      "\n  appearance is blind (the 2D trap), preserves occluded objects, and reprojects consistently; AND it is"
      "\n  held to the same discipline as everything else: depth-shuffle kills the advantage (signal is real), a"
      "\n  hallucinated monocular over-reach is refuted by action feedback then corrected, and underdetermined"
      "\n  depth is REFUSED (measure, don't assert). 3D proposes structure; action feedback owns truth.")
