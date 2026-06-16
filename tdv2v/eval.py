"""Online KNN probe (the paper's quality + collapse proxy) on the [CLS] embedding. On synthetic data the
label is object identity, so KNN measures whether reps separate objects = the semantic axis TDV is weak on.
Swap the labeled set for ImageNet on the real runs."""
import torch
from torch.utils.data import DataLoader
from data import SyntheticStereoVideo


@torch.no_grad()
def knn_eval(encoder, device, img=224, n_classes=20, n=600, k=20, seed=999):
    enc = encoder.eval()
    ds = SyntheticStereoVideo(img=img, n_classes=n_classes, epoch_len=n, seed=seed)
    dl = DataLoader(ds, batch_size=64, num_workers=0)
    feats, labels = [], []
    for A_t, _, _, _, y in dl:                      # probe camera-A frame at time t
        z = enc(A_t.to(device))[:, 0]               # [CLS]
        feats.append(torch.nn.functional.normalize(z, dim=1).cpu()); labels.append(y)
    F = torch.cat(feats); Y = torch.cat(labels)
    ntr = int(0.7 * len(Y)); idx = torch.randperm(len(Y), generator=torch.Generator().manual_seed(seed))
    tr, te = idx[:ntr], idx[ntr:]
    sim = F[te] @ F[tr].T                           # cosine (normalized)
    nn_idx = sim.topk(k, dim=1).indices
    pred = Y[tr][nn_idx].mode(dim=1).values
    acc = (pred == Y[te]).float().mean().item()
    return acc                                       # chance = 1 / n_classes
