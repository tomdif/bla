"""Two-view TDV pretrainer. One script, flags select the head-to-head condition:
  monocular TDV ............ (no flags)            temporal prediction only
  + cross-view ............. --crossview           + CroCo-style cross-view completion
  + commutativity .......... --crossview --commut  + (time×view) cycle consistency
  shuffle control .......... --crossview --shuffle-control   break A–B pairing (causal control)
Anti-collapse: --anticollapse sigreg (default) | dino.
"""
import argparse, os, math, json, time, torch
from torch.utils.data import DataLoader
from model import ViT, MotionEncoder, CrossViewHead, ProjHead, make_teacher, ema_update
from losses import temporal_mse, crossview_mse, commutativity, sigreg, dino, effective_rank
from data import build_dataset
from eval import knn_eval


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="synthetic"); p.add_argument("--root", default="")
    p.add_argument("--img", type=int, default=224); p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dim", type=int, default=384); p.add_argument("--depth", type=int, default=12)
    p.add_argument("--heads", type=int, default=6); p.add_argument("--motion-depth", type=int, default=4)
    p.add_argument("--xview-depth", type=int, default=4); p.add_argument("--n-classes", type=int, default=20)
    p.add_argument("--crossview", action="store_true"); p.add_argument("--commut", action="store_true")
    p.add_argument("--shuffle-control", action="store_true")
    p.add_argument("--anticollapse", default="sigreg", choices=["sigreg", "dino"])
    p.add_argument("--w-temporal", type=float, default=1.0); p.add_argument("--w-xview", type=float, default=1.0)
    p.add_argument("--w-commut", type=float, default=0.5); p.add_argument("--w-ac", type=float, default=1.0)
    p.add_argument("--bs", type=int, default=64); p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=0.05); p.add_argument("--ema", type=float, default=0.996)
    p.add_argument("--steps", type=int, default=5000); p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--log-every", type=int, default=50); p.add_argument("--knn-every", type=int, default=500)
    p.add_argument("--workers", type=int, default=4); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/exp"); p.add_argument("--device", default="")
    return p.parse_args()


def lr_at(step, args):
    if step < args.warmup: return args.lr * step / max(1, args.warmup)
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return 0.5 * args.lr * (1 + math.cos(math.pi * t))


def main():
    a = get_args(); torch.manual_seed(a.seed); os.makedirs(a.out, exist_ok=True)
    dev = a.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device={dev}  condition: crossview={a.crossview} commut={a.commut} "
          f"shuffle={a.shuffle_control} anticollapse={a.anticollapse}")

    f = ViT(a.img, a.patch, a.dim, a.depth, a.heads).to(dev)            # student frame encoder
    teacher = make_teacher(f).to(dev)                                   # EMA teacher
    motion = MotionEncoder(a.img, a.patch, a.dim, a.heads, a.motion_depth).to(dev)
    n_tok = f.embed.n + 1
    xview = CrossViewHead(n_tok, a.dim, a.heads, a.xview_depth).to(dev) if a.crossview else None
    params = list(f.parameters()) + list(motion.parameters()) + (list(xview.parameters()) if xview else [])
    proj = proj_t = center = None
    if a.anticollapse == "dino":
        proj = ProjHead(a.dim).to(dev); proj_t = make_teacher(proj).to(dev)
        center = torch.zeros(proj.net[-1].out_features, device=dev); params += list(proj.parameters())
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=a.wd)

    kw = dict(n_classes=a.n_classes, epoch_len=a.bs * a.log_every * 20) if a.data == "synthetic" else dict(root=a.root)
    dl = DataLoader(build_dataset(a.data, a.img, **kw), batch_size=a.bs, shuffle=True,
                    num_workers=a.workers, drop_last=True, persistent_workers=a.workers > 0)

    step = 0; t0 = time.time(); it = iter(dl)
    while step < a.steps:
        try: A_t, A_t1, B_t, B_t1, y = next(it)
        except StopIteration: it = iter(dl); A_t, A_t1, B_t, B_t1, y = next(it)
        A_t, A_t1, B_t, B_t1 = (x.to(dev, non_blocking=True) for x in (A_t, A_t1, B_t, B_t1))
        if a.shuffle_control:                                          # break A–B correspondence
            perm = torch.randperm(B_t.size(0), device=dev); B_t, B_t1 = B_t[perm], B_t1[perm]
        for g in opt.param_groups: g["lr"] = lr_at(step, a)

        zA_t, zB_t = f(A_t), f(B_t)                                     # student
        with torch.no_grad():
            tzA_t1, tzB_t1, tzB_t = teacher(A_t1), teacher(B_t1), teacher(B_t)
        # --- temporal (TDV) on both cameras ---
        dzA = motion(A_t1 - A_t, zA_t); dzB = motion(B_t1 - B_t, zB_t)
        L_temp = temporal_mse(zA_t + dzA, tzA_t1) + temporal_mse(zB_t + dzB, tzB_t1)
        # --- cross-view completion ---
        L_xv = crossview_mse(xview(zA_t), tzB_t) if a.crossview else torch.zeros((), device=dev)
        # --- commutativity (time × view) ---
        if a.commut:
            zA_t1, zB_t1 = f(A_t1), f(B_t1)
            L_co = commutativity(dzA, zB_t1 - zA_t1, zB_t - zA_t, dzB)
        else:
            L_co = torch.zeros((), device=dev)
        # --- anti-collapse on [CLS] of student frames ---
        emb = torch.cat([zA_t[:, 0], zB_t[:, 0]], 0)
        if a.anticollapse == "sigreg":
            L_ac = sigreg(emb)
        else:
            with torch.no_grad():
                temb = torch.cat([teacher(A_t)[:, 0], tzB_t[:, 0]], 0); tp = proj_t(temb)
            L_ac = dino(proj(emb), tp, center); center.mul_(0.9).add_(0.1 * tp.mean(0))

        loss = (a.w_temporal * L_temp + a.w_xview * L_xv + a.w_commut * L_co + a.w_ac * L_ac)
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 3.0); opt.step()
        ema_update(f, teacher, a.ema)
        if a.anticollapse == "dino": ema_update(proj, proj_t, a.ema)

        if step % a.log_every == 0:
            er = effective_rank(emb.detach())
            print(f"[{step:6d}] loss={loss.item():.4f} temp={L_temp.item():.4f} xview={L_xv.item():.4f} "
                  f"commut={L_co.item():.4f} ac={L_ac.item():.4f} | eff_rank={er:.1f} "
                  f"({(step+1)*a.bs/(time.time()-t0):.0f} img/s)")
        if step > 0 and step % a.knn_every == 0:
            acc = knn_eval(teacher, dev, a.img, a.n_classes)
            print(f"   >>> KNN acc (object-id; chance={1/a.n_classes:.3f}) = {acc:.3f}")
            json.dump({"step": step, "knn": acc, "eff_rank": effective_rank(emb.detach())},
                      open(f"{a.out}/metrics_{step}.json", "w"))
        step += 1

    torch.save({"frame": f.state_dict(), "teacher": teacher.state_dict()}, f"{a.out}/ckpt.pt")
    print(f"final KNN = {knn_eval(teacher, dev, a.img, a.n_classes):.3f}  -> saved {a.out}/ckpt.pt")


if __name__ == "__main__":
    main()
