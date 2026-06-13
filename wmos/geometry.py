"""GeometryCanvas -- an object-centric 2.5D layer (pose + depth uncertainty), NOT a dense reconstruction.

Validated controls-first in geometry_canvas_gate.py. Same rule as the rest of WMOS: 3D PROPOSES
structure (object pose / depth via triangulation); action feedback OWNS truth. A single view loses
depth (a perspective ray is ambiguous) -> high sigma -> refuse; a second view (stereo/motion)
triangulates it -> low sigma -> assert. Pure stdlib (keeps the package zero-dependency).
"""
import math


def project(p, cam=(0.0, 0.0, 0.0)):                 # pinhole, focal=1, looking +z
    q = (p[0] - cam[0], p[1] - cam[1], p[2] - cam[2])
    z = q[2] if abs(q[2]) > 1e-9 else 1e-9
    return (q[0] / z, q[1] / z)


def dist3(p): return math.sqrt(p[0] * p[0] + p[1] * p[1] + p[2] * p[2])


class GeometryCanvas:
    def __init__(self, reach=1.5, baseline=0.3, sigma_ood=0.5, mono_prior_z=1.0):
        self.reach = reach; self.baseline = baseline; self.sigma_ood = sigma_ood; self.mono_prior_z = mono_prior_z

    def estimate_one(self, p_world, stereo=True):
        uv0 = project(p_world)
        if stereo:
            uv1 = project(p_world, (self.baseline, 0, 0))
            disp = uv0[0] - uv1[0]
            z = self.baseline / disp if abs(disp) > 1e-9 else 1e6
            sigma = 0.05
        else:
            z = self.mono_prior_z; sigma = 0.6                # depth underdetermined from one view
        est = (uv0[0] * z, uv0[1] * z, z)
        ood = sigma > self.sigma_ood
        reach_pred = None if ood else (dist3(est) + sigma <= self.reach)   # refuse (None) when underdetermined
        # reprojection residual into both views (consistency)
        re = math.hypot(*(a - b for a, b in zip(project(est), uv0)))
        if stereo:
            re += math.hypot(*(a - b for a, b in zip(project(est, (self.baseline, 0, 0)), project(p_world, (self.baseline, 0, 0)))))
        return {"pose3d": tuple(round(v, 3) for v in est), "depth": round(est[2], 3), "depth_sigma": sigma,
                "est_dist": round(dist3(est), 3), "reach_pred": reach_pred, "ood": ood,
                "reproj_err": round(re, 4), "uv": (round(uv0[0], 3), round(uv0[1], 3))}

    def estimate(self, objs_world, stereo=True):
        return {oid: self.estimate_one(p, stereo) for oid, p in objs_world.items()}
