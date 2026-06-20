import torch, numpy as np, time
torch.manual_seed(2); np.random.seed(2)
# RICHER affordance world: movable object with INVISIBLE affordance operator M(tau)=g*R(theta) (Δp=M·a, BILINEAR),
# + immovable distractor objects (action-independent). tau=(theta,g) inferable from one (Δp,a_prev) pair.
# Compositional transfer: train theta in [0,pi], TEST held-out theta in [pi,2pi].
def gen(th_lo,th_hi,n_ep=1500,T=40,D=2,sig=0.2,seed=0):
    rng=np.random.RandomState(seed)
    th=rng.uniform(th_lo,th_hi,(n_ep,1)); g=rng.uniform(0.6,1.4,(n_ep,1))
    c,s=np.cos(th),np.sin(th)
    M=np.stack([np.concatenate([g*c,-g*s],1), np.concatenate([g*s,g*c],1)],1)  # (n,2,2)
    Mcol0=np.concatenate([g*c,g*s],1)                                          # affordance probe target (2D)
    p=rng.randn(n_ep,2); pprev=p.copy(); aprev=np.zeros((n_ep,2)); d=0.7*rng.randn(n_ep,2*D)
    Wf=[];Wn=[];A=[];Aff=[]
    for t in range(T):
        a=rng.randn(n_ep,2)
        feat=np.concatenate([p,pprev,aprev,d,0.9*d],1)+sig*rng.randn(n_ep,6+4*D)
        dp=np.einsum('nij,nj->ni',M,a)                                        # Δp = M·a  (bilinear affordance)
        npn=0.95*p+dp; nd=0.9*d+0.7*rng.randn(n_ep,2*D)
        nxt=np.concatenate([npn,d,p],1)+sig*rng.randn(n_ep,4+2*D)             # target: next movable p + dist
        Wf.append(feat);Wn.append(nxt);A.append(a);Aff.append(Mcol0.copy())
        pprev=p;aprev=a;p=npn;d=nd
    f=lambda v: torch.tensor(np.concatenate(v),dtype=torch.float32)
    return f(Wf),f(Wn),f(A),f(Aff)
class Enc(torch.nn.Module):
    def __init__(s,din,k): super().__init__(); s.net=torch.nn.Sequential(torch.nn.Linear(din,128),torch.nn.SiLU(),torch.nn.Linear(128,k))
    def forward(s,x): return s.net(x)
def train_wm(W,A,Wn,k,mode,steps=4000,da=2):
    din=W.shape[1]; tdim=Wn.shape[1]; enc=Enc(din,k); P=list(enc.parameters())
    if mode=='concat':
        h=torch.nn.Sequential(torch.nn.Linear(k+da,128),torch.nn.SiLU(),torch.nn.Linear(128,tdim)); P+=list(h.parameters()); fwd=lambda z,a:h(torch.cat([z,a],1))
    else:
        ae=2; base=torch.nn.Linear(k,tdim); Tn=torch.nn.Sequential(torch.nn.Linear(da,16),torch.nn.SiLU(),torch.nn.Linear(16,ae))
        hy=torch.nn.Sequential(torch.nn.Linear(k,128),torch.nn.SiLU(),torch.nn.Linear(128,tdim*ae+tdim)); P+=list(base.parameters())+list(Tn.parameters())+list(hy.parameters())
        def fwd(z,a):
            hh=hy(z); Wm=hh[:,:tdim*ae].view(-1,tdim,ae); b=hh[:,tdim*ae:]
            return base(z)+torch.bmm(Wm,Tn(a).unsqueeze(-1)).squeeze(-1)+b
    opt=torch.optim.Adam(P,2e-3); N=W.shape[0]
    for it in range(steps):
        i=torch.randint(0,N,(512,)); loss=((fwd(enc(W[i]),A[i])-Wn[i])**2).mean(); opt.zero_grad();loss.backward();opt.step()
    return enc,fwd
def pred_mse_p(enc,fwd,W,A,Wn):                         # MSE on the MOVABLE object's next-pos (first 2 dims)
    with torch.no_grad(): return float(((fwd(enc(W),A)[:,:2]-Wn[:,:2])**2).mean())
def r2(enc,W,Y):
    with torch.no_grad(): L=enc(W).numpy()
    y=Y.numpy(); n=int(0.7*len(y)); X=np.concatenate([L[:n],np.ones((n,1))],1); wv,*_=np.linalg.lstsq(X,y[:n],rcond=None)
    p=np.concatenate([L[n:],np.ones((len(y)-n,1))],1)@wv; yt=y[n:]; return float(max(0.,1-((yt-p)**2).sum()/((yt-yt.mean(0))**2).sum()))
t0=time.time()
Wtr,Wntr,Atr,Afftr=gen(0,np.pi,seed=0)                 # train affordances
Wte,Wnte,Ate,Affte=gen(np.pi,2*np.pi,seed=9)           # HELD-OUT affordances (unseen rotation directions)
print("RICHER affordance: Δp=M(tau)·a bilinear; train theta in [0,pi], TEST held-out [pi,2pi]")
print(f"{'mode':9} | {'pred-MSE in-dist':>16} {'pred-MSE HELD-OUT':>17} | {'afford-decode held-out':>22}")
for mode in ['concat','operator']:
    enc,fwd=train_wm(Wtr,Atr,Wntr,8,mode)
    mi=pred_mse_p(enc,fwd,Wtr,Atr,Wntr); mo=pred_mse_p(enc,fwd,Wte,Ate,Wnte); af=r2(enc,Wte,Affte)
    print(f"{mode:9} | {mi:>16.3f} {mo:>17.3f} | {af:>22.2f}")
print("\nPRED: operator HELD-OUT MSE << concat -> bilinear bias extrapolates = operator earns its place.")
print("      affordance-decode operator > concat -> the affordance field transfers.  (tie => re-param, drop it)")
print(f"{time.time()-t0:.0f}s")
