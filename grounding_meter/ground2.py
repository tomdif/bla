import torch, numpy as np, time
torch.manual_seed(0); np.random.seed(0)

def gen(alpha, ctrl, n_ep=800, T=50, D=2, sigma=0.35, dg=0.4, xg=1.2):
    sx=np.random.randn(n_ep,1); sz=np.random.randn(n_ep,1); sd=dg*np.random.randn(n_ep,D)
    Ot=[];Otp=[];Aa=[];Zt=[];Xt=[]
    for t in range(T):
        ax=np.random.randn(n_ep,1); az=np.random.randn(n_ep,1)
        ot=np.concatenate([xg*sx, alpha*sz, sd],1)+sigma*np.random.randn(n_ep,2+D)
        zt=sz.copy(); xt=sx.copy()
        nx=0.95*sx+ax; nz=0.95*sz+(az if ctrl else np.random.randn(n_ep,1)); nd=0.95*sd+dg*np.random.randn(n_ep,D)
        otp=np.concatenate([xg*nx, alpha*nz, nd],1)+sigma*np.random.randn(n_ep,2+D)
        Ot.append(ot);Otp.append(otp);Aa.append(np.concatenate([ax,az],1));Zt.append(zt);Xt.append(xt)
        sx,sz,sd=nx,nz,nd
    f=lambda v: torch.tensor(np.concatenate(v),dtype=torch.float32)
    return f(Ot),f(Otp),f(Aa),f(Zt),f(Xt)

class Enc(torch.nn.Module):
    def __init__(s,din,k): super().__init__(); s.net=torch.nn.Sequential(torch.nn.Linear(din,64),torch.nn.ReLU(),torch.nn.Linear(64,k))
    def forward(s,x): return s.net(x)

def train_arm(O,Op,A,k,arm,steps=1000):
    din=O.shape[1]; enc=Enc(din,k)
    head=(torch.nn.Sequential(torch.nn.Linear(k+A.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,din)) if arm=='pred'
          else torch.nn.Sequential(torch.nn.Linear(2*k,64),torch.nn.ReLU(),torch.nn.Linear(64,A.shape[1])))
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),1e-3); N=O.shape[0]
    for it in range(steps):
        idx=torch.randint(0,N,(512,))
        pred=(head(torch.cat([enc(O[idx]),A[idx]],1)) if arm=='pred' else head(torch.cat([enc(O[idx]),enc(Op[idx])],1)))
        tgt=(Op[idx] if arm=='pred' else A[idx]); loss=((pred-tgt)**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    return enc

def r2(enc,O,Y,kind='lin'):
    with torch.no_grad(): L=enc(O).numpy()
    y=Y.numpy().ravel(); ntr=int(0.7*len(y))
    if kind=='lin':
        Xtr=np.concatenate([L[:ntr],np.ones((ntr,1))],1); w,*_=np.linalg.lstsq(Xtr,y[:ntr],rcond=None)
        pred=np.concatenate([L[ntr:],np.ones((len(y)-ntr,1))],1)@w
    else:
        m=torch.nn.Sequential(torch.nn.Linear(L.shape[1],64),torch.nn.ReLU(),torch.nn.Linear(64,1)); o=torch.optim.Adam(m.parameters(),1e-2)
        Lt=torch.tensor(L[:ntr]);yt=torch.tensor(y[:ntr,None],dtype=torch.float32)
        for _ in range(400): o.zero_grad();l=((m(Lt)-yt)**2).mean();l.backward();o.step()
        with torch.no_grad(): pred=m(torch.tensor(L[ntr:])).numpy().ravel()
    yte=y[ntr:]; return max(0.,1-((yte-pred)**2).sum()/((yte-yte.mean())**2).sum())

t0=time.time()
print("alpha |  CONTROLLABLE: predZ effZ  (predZ_MLP) [xsan]  | EXOGENOUS: predZ effZ")
for a in [0.0,0.05,0.1,0.15,0.2,0.3,0.4,0.5,0.7,1.0]:
    out=[]
    for ctrl in [True,False]:
        O,Op,A,Z,X=gen(a,ctrl); ep=train_arm(O,Op,A,2,'pred'); ee=train_arm(O,Op,A,2,'eff')
        out.append((r2(ep,O,Z),r2(ee,O,Z),r2(ep,O,Z,'mlp'),r2(ep,O,X)))
    c,e=out
    print(f"{a:<5} |  predZ={c[0]:.2f} effZ={c[1]:.2f}  (MLP={c[2]:.2f}) [x={c[3]:.2f}]  | predZ={e[0]:.2f} effZ={e[1]:.2f}")
print(f"{time.time()-t0:.0f}s")
