import torch, numpy as np, time
torch.manual_seed(1); np.random.seed(1)
# STEELMAN keystone regime: c = INVISIBLE config (constant/episode), scales the action's effect on a
# VISIBLE position p. Rollout (no a_t) can't need c; concat/operator (a_t known) must infer c to predict.
def gen(n_ep=1200, T=40, D=2, sig=0.25, dg=0.8):
    c=np.random.randn(n_ep,1)                          # invisible config (episode-constant gain)
    p=np.random.randn(n_ep,1); pa=np.zeros((n_ep,1)); pprev=p.copy(); aprev=np.zeros((n_ep,1))
    sd=dg*np.random.randn(n_ep,D)
    W=[];Wn=[];A=[];Cc=[]   # window feat, next-window-target, action, config-gt
    for t in range(T):
        a=np.random.randn(n_ep,1)
        # encoder window: [p_t, p_{t-1}, a_{t-1}, d_t, d_{t-1}] -> c inferable from (p_t-p_{t-1})/a_{t-1}
        feat=np.concatenate([p, pprev, aprev, sd, 0.9*sd],1)+sig*np.random.randn(n_ep,3+2*D)
        gain=0.5+0.6*c                                  # action effect scaled by invisible config
        npn=p+gain*a; nd=0.9*sd+dg*np.random.randn(n_ep,D)
        nextfeat=np.concatenate([npn, p, a, nd, sd],1)+sig*np.random.randn(n_ep,3+2*D)  # target obs
        W.append(feat);Wn.append(nextfeat);A.append(a);Cc.append(c.copy())
        pprev=p; aprev=a; p=npn; sd=nd
    f=lambda v: torch.tensor(np.concatenate(v),dtype=torch.float32)
    return f(W),f(Wn),f(A),f(Cc)

class Enc(torch.nn.Module):
    def __init__(s,din,k): super().__init__(); s.net=torch.nn.Sequential(torch.nn.Linear(din,96),torch.nn.SiLU(),torch.nn.Linear(96,k))
    def forward(s,x): return s.net(x)
def train_wm(W,A,Wn,k,mode,steps=3000,da=1):
    din=W.shape[1]; enc=Enc(din,k); P=list(enc.parameters())
    if mode=='rollout':
        h=torch.nn.Sequential(torch.nn.Linear(k,96),torch.nn.SiLU(),torch.nn.Linear(96,din)); P+=list(h.parameters()); fwd=lambda z,a:h(z)
    elif mode=='concat':
        h=torch.nn.Sequential(torch.nn.Linear(k+da,96),torch.nn.SiLU(),torch.nn.Linear(96,din)); P+=list(h.parameters()); fwd=lambda z,a:h(torch.cat([z,a],1))
    else:
        ae=8; base=torch.nn.Linear(k,din); Tn=torch.nn.Sequential(torch.nn.Linear(da,32),torch.nn.SiLU(),torch.nn.Linear(32,ae))
        hy=torch.nn.Sequential(torch.nn.Linear(k,96),torch.nn.SiLU(),torch.nn.Linear(96,din*ae+din)); P+=list(base.parameters())+list(Tn.parameters())+list(hy.parameters())
        def fwd(z,a):
            hh=hy(z); Wm=hh[:,:din*ae].view(-1,din,ae); b=hh[:,din*ae:]
            return base(z)+torch.bmm(Wm,Tn(a).unsqueeze(-1)).squeeze(-1)+b
    opt=torch.optim.Adam(P,2e-3); N=W.shape[0]
    for it in range(steps):
        i=torch.randint(0,N,(512,)); loss=((fwd(enc(W[i]),A[i])-Wn[i])**2).mean(); opt.zero_grad();loss.backward();opt.step()
    return enc
def r2(enc,W,Y):
    with torch.no_grad(): L=enc(W).numpy()
    y=Y.numpy(); n=int(0.7*len(y)); X=np.concatenate([L[:n],np.ones((n,1))],1); wv,*_=np.linalg.lstsq(X,y[:n],rcond=None)
    p=np.concatenate([L[n:],np.ones((len(y)-n,1))],1)@wv; yt=y[n:]; return float(max(0.,1-((yt-p)**2).sum()/((yt-yt.mean(0))**2).sum()))

t0=time.time(); W,Wn,A,C=gen()
print("STEELMAN: c=INVISIBLE config scaling action effect; window lets c be INFERRED from (Δp / a_prev)")
print(f"{'k':>2} | {'rollout c':>9} {'concat c':>9} {'operator c':>10} | gap(concat-rollout)")
for k in [3,4,6]:
    rc=r2(train_wm(W,A,Wn,k,'rollout'),W,C); cc=r2(train_wm(W,A,Wn,k,'concat'),W,C); oc=r2(train_wm(W,A,Wn,k,'operator'),W,C)
    print(f"{k:>2} | {rc:>9.2f} {cc:>9.2f} {oc:>10.2f} | {cc-rc:+.2f}")
print(f"{time.time()-t0:.0f}s")
