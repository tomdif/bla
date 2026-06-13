"""Safety layer -- the meta-defenses the adversarial red-team (adversarial_gate.py) named for the three
statistical breaches. Each closes one hole; each still rests on a stated assumption (you must monitor
the features, you must detect the shift, the risk must be made observable).

  CompleteOODDetector   closes OOD-EVASION: flag if ANY monitored feature is out of range (not just one).
  ShiftDetector         closes CONFORMAL-UNDER-SHIFT: refuse to trust a calibrated band when the live
                        batch's feature distribution has drifted from calibration (exchangeability broken).
  irreversible_unknown_risk  closes DISGUISED-TRAP: never take an IRREVERSIBLE action whose risk is
                        unobservable -- defer/measure instead of committing on a prediction.
Pure stdlib.
"""
import math


def _numeric_keys(feat):
    return [k for k, v in feat.items() if isinstance(v, (int, float)) and not isinstance(v, bool)]


class CompleteOODDetector:
    """OOD if ANY calibrated feature is out of its range. Monitor every feature that can drive the label."""
    def __init__(self, z=2.5): self.calib = {}; self.z = z
    def calibrate(self, feat_dicts, keys=None):
        if not feat_dicts: return self
        keys = keys or _numeric_keys(feat_dicts[0])
        for k in keys:
            vals = [f[k] for f in feat_dicts if k in f]
            if not vals: continue
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1e-6
            self.calib[k] = (mu, sd, min(vals), max(vals))
        return self
    def check(self, feat):
        for k, (mu, sd, lo, hi) in self.calib.items():
            if k in feat and abs((feat[k] - mu) / sd) > self.z:
                return True, k                       # the offending feature
        return False, None


class ShiftDetector:
    """Detect distribution shift between calibration and a live batch -> conformal coverage not trustworthy."""
    def __init__(self, thresh=1.0): self.ref = {}; self.thresh = thresh
    def fit(self, feat_dicts, keys=None):
        keys = keys or _numeric_keys(feat_dicts[0])
        for k in keys:
            vals = [f[k] for f in feat_dicts if k in f]
            mu = sum(vals) / len(vals)
            sd = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1e-6
            self.ref[k] = (mu, sd)
        return self
    def score(self, batch):
        s = 0.0
        for k, (mu, sd) in self.ref.items():
            vals = [f[k] for f in batch if k in f]
            if vals: s = max(s, abs(sum(vals) / len(vals) - mu) / sd)
        return s
    def shifted(self, batch): return self.score(batch) >= self.thresh


def irreversible_unknown_risk(feat):
    """True when an action is irreversible AND its risk is unobservable -> must not act on a prediction."""
    return bool(feat.get("irreversible")) and not bool(feat.get("risk_observable", True))
