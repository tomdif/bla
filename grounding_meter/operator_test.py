import torch, numpy as np, time
torch.manual_seed(0); np.random.seed(0)
# Controlled world: c=controllable(quiet), u=uncontrolled(quiet, decision-relevant via reward but NO return here),
# d=loud exogenous distractors. Bottleneck encoder must COMPETE. Action moves c (config-dependent gain).
def gen(n_ep=900, T=50, D=4, sig=0.3, ac=0.7, au=0.0, dg=0.7):
    sc=np.random.randn(n_ep,2); su=np.random.randn(n_ep,2); sd=dg*np.random.randn(n_ep,D)
    Ot=[];Otp=[];A=[];Ct=[];Ut=[]
    for t in range(T):
        a=np.random.randn(n_ep,2)
        o=np.concatenate([ac*sc, au*su, sd],1)+sig*np.random.randn(n_ep,4+D)
        ct=sc.copy(); ut=su.copy()
        gain=1.0+0.5*np.tanh(sc)                       # config-DEPENDENT action effect (operator's home turf)
        nc=0.5*sc+2.0*gain*a; nu=0.9*su+np.random.randn(n_ep,2); nd=0.9*sd+dg*np.random.randn(n_ep,D)  # c: weak persistence, LARGE action authority
        otp=np.concatenate([ac*nc, au*nu, nd],1)+sig*np.random.randn(n_ep,4+D)
        Ot.append(o);Otp.append(otp);A.append(a);Ct.append(ct);Ut.append(ut); sc,su,sd=nc,nu,nd
    f=lambda v: torch.tensor(np.concatenate(v),dtype=torch.float32)
    return f(Ot),f(Otp),f(A),f(Ct),f(Ut)

class Enc(torch.nn.Module):
    def __init__(s,din,k): super().__init__(); s.net=torch.nn.Sequential(torch.nn.Linear(din,64),torch.nn.SiLU(),torch.nn.Linear(64,k))
    def forward(s,x): return s.net(x)

def train_wm(O,A,Otp1,k,mode,steps=1500,da=2):
    din=O.shape[1]; enc=Enc(din,k); params=list(enc.parameters())
    if mode=='rollout':
        head=torch.nn.Sequential(torch.nn.Linear(k,64),torch.nn.SiLU(),torch.nn.Linear(64,din)); params+=list(head.parameters())
        fwd=lambda z,a: head(z)
    elif mode=='concat':
        head=torch.nn.Sequential(torch.nn.Linear(k+da,64),torch.nn.SiLU(),torch.nn.Linear(64,din)); params+=list(head.parameters())
        fwd=lambda z,a: head(torch.cat([z,a],1))
    else: # operator: Omega=Hyper(z) emits affordance W(z),b(z); T(a) embeds action; pred = base(z)+W(z)@T(a)
        ae=8; base=torch.nn.Linear(k,din); Tnet=torch.nn.Sequential(torch.nn.Linear(da,32),torch.nn.SiLU(),torch.nn.Linear(32,ae))
        hyper=torch.nn.Sequential(torch.nn.Linear(k,64),torch.nn.SiLU(),torch.nn.Linear(64,din*ae+din))
        params+=list(base.parameters())+list(Tnet.parameters())+list(hyper.parameters())
        def fwd(z,a):
            h=hyper(z); W=h[:,:din*ae].view(-1,din,ae); b=h[:,din*ae:]
            return base(z)+torch.bmm(W, Tnet(a).unsqueeze(-1)).squeeze(-1)+b
    opt=torch.optim.Adam(params,2e-3); N=O.shape[0]
    for it in range(steps):
        idx=torch.randint(0,N,(512,)); loss=((fwd(enc(O[idx]),A[idx])-Otp1[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    return enc

def r2(enc,O,Y):
    with torch.no_grad(): L=enc(O).numpy()
    y=Y.numpy(); n=int(0.7*len(y)); Xtr=np.concatenate([L[:n],np.ones((n,1))],1)
    W,*_=np.linalg.lstsq(Xtr,y[:n],rcond=None); p=np.concatenate([L[n:],np.ones((len(y)-n,1))],1)@W; yt=y[n:]
    return float(max(0.,1-((yt-p)**2).sum()/((yt-yt.mean(0))**2).sum()))

t0=time.time(); O,Otp1,A,C,U=gen()
print(f"world: obs_dim={O.shape[1]} | c=controllable(visible,weak-persistence,big-action) | u=uncontrolled INVISIBLE(residue) | d=4 loud distractors | bottleneck k=3 < #encodables")
print(f"{'mode':9} | {'c_decode(controllable)':>22} | {'u_decode(uncontrolled/residue)':>30}")
res={}
for mode in ['rollout','concat','operator']:
    enc=train_wm(O,A,Otp1,3,mode,steps=2500); cc=r2(enc,O,C); uu=r2(enc,O,U); res[mode]=(cc,uu)
    print(f"{mode:9} | {cc:>22.2f} | {uu:>30.2f}")
print(f"\nP1 keystone: rollout c={res['rollout'][0]:.2f} (should ~0) vs concat {res['concat'][0]:.2f} / operator {res['operator'][0]:.2f} (should be high)")
print(f"P2 op-vs-concat: operator {res['operator'][0]:.2f} vs concat {res['concat'][0]:.2f} (tie=>re-param; op>concat=>real bias)")
print(f"P3 residue: u-decode all low: rollout {res['rollout'][1]:.2f} concat {res['concat'][1]:.2f} operator {res['operator'][1]:.2f} (no return => ungrounded)")
print(f"{time.time()-t0:.0f}s")
