"""Datasets. `SyntheticStereoVideo` runs out-of-the-box (smoke test + controlled ablation) and emits the
four grid frames (A,t), (A,t+1), (B,t), (B,t+1) plus an object-identity label for the KNN probe.
Swap in `SceneFlowVideo` / `KITTIStereoVideo` (stubs below) for the real Tier-1 runs."""
import numpy as np, torch
from torch.utils.data import Dataset


def _obj_props(cls):
    r = np.random.default_rng(10_000 + cls)
    return r.integers(0, 255, (64, 64, 3)), r.integers(0, 160, (3,)), int(r.integers(0, 2))  # tex,color,shape


def _render(rng, pos, ap, tex, color, shape, H, W):
    base = int(rng.integers(70, 170))
    im = np.full((H, W, 3), base, np.uint8)
    im = (im + rng.integers(-30, 30, (H, W, 3))).clip(0, 255).astype(np.uint8)   # per-view background
    cx, cy = pos
    for yy in range(ap):
        for xx in range(ap):
            if shape == 1 and (xx - ap / 2) ** 2 + (yy - ap / 2) ** 2 > (ap / 2) ** 2:
                continue
            X, Y = cx - ap // 2 + xx, cy - ap // 2 + yy
            if 0 <= X < W and 0 <= Y < H:
                im[Y, X] = (tex[yy % 64, xx % 64] * 0.5 + color).clip(0, 255)
    return im


def _to_tensor(im):
    return torch.from_numpy(im).permute(2, 0, 1).float().div(255).sub(0.5).div(0.5)


class SyntheticStereoVideo(Dataset):
    """Two cameras (horizontal baseline + scale/shear), two times (object moves). Object identity is the
    label. `epoch_len` controls samples/epoch (generated on the fly)."""
    def __init__(self, img=224, n_classes=20, epoch_len=2000, seed=0):
        self.H = self.W = img; self.n_classes = n_classes; self.len = epoch_len; self.seed = seed

    def __len__(self): return self.len

    def __getitem__(self, idx):
        rng = np.random.default_rng(self.seed * 1_000_003 + idx)
        cls = int(rng.integers(0, self.n_classes)); tex, color, shape = _obj_props(cls)
        d = rng.uniform(2.0, 9.0); ap = int(np.clip(56.0 / d * 3.0, 16, 80))
        disp = int(np.clip(150.0 / d, 12, 64))
        ax, ay = int(rng.integers(70, 150)), int(rng.integers(70, 150))
        mvx, mvy = int(rng.integers(-14, 14)), int(rng.integers(-10, 10))     # temporal motion
        sh = 0.15
        def shift(p, dx): return (int(np.clip(p[0] + dx, 10, self.W - 10)), p[1])
        A_t = _render(rng, (ax, ay), ap, tex, color, shape, self.H, self.W)
        A_t1 = _render(rng, (ax + mvx, ay + mvy), ap, tex, color, shape, self.H, self.W)
        B_t = _render(rng, shift((ax, ay), -disp), int(ap * 0.85), tex, color, shape, self.H, self.W)
        B_t1 = _render(rng, shift((ax + mvx, ay + mvy), -disp), int(ap * 0.85), tex, color, shape, self.H, self.W)
        return (_to_tensor(A_t), _to_tensor(A_t1), _to_tensor(B_t), _to_tensor(B_t1), cls)


class SceneFlowVideo(Dataset):
    """STUB: FlyingThings3D / SceneFlow stereo video (synthetic, GT depth+flow; already in the TDV eval
    pipeline). Return (A_t, A_t1, B_t, B_t1, label_or_dummy). Point `root` at the SceneFlow frames."""
    def __init__(self, root, img=224):
        raise NotImplementedError("Wire SceneFlow frames here: left/right = cameras A/B, consecutive = t/t+1.")


class KITTIStereoVideo(Dataset):
    """STUB: KITTI raw stereo driving video (real, calibrated). Same 4-frame contract as above."""
    def __init__(self, root, img=224):
        raise NotImplementedError("Wire KITTI raw stereo sequences here (image_02 = A, image_03 = B).")


def build_dataset(name, img, **kw):
    if name == "synthetic": return SyntheticStereoVideo(img=img, **kw)
    if name == "sceneflow": return SceneFlowVideo(kw["root"], img=img)
    if name == "kitti": return KITTIStereoVideo(kw["root"], img=img)
    raise ValueError(name)
