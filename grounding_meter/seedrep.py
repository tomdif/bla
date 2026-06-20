import torch, numpy as np
exec(open('/tmp/usable3.py').read().split('t0=time.time()')[0])  # reuse defs
res={'lin':[],'mlp':[]}
for seed in range(4):
    torch.manual_seed(seed); np.random.seed(seed)
    enc,dl,dm=pretrain(0.0)
    for cap in ['lin','mlp']:
        eo=runpol('oracle',cap);eb=runpol('blind',cap);el=runpol('latent',cap,enc,0.0)
        res[cap].append((eb-el)/(eb-eo+1e-9))
for cap in ['lin','mlp']:
    a=np.array(res[cap]); print(f"clean band-1 usability ({cap}-policy): {a.mean():.2f} +/- {a.std():.2f}   seeds={np.round(a,2).tolist()}")
