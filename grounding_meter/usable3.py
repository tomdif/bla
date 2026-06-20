import torch, numpy as np, time
torch.manual_seed(0); np.random.seed(0)
ALPHA=1.0; SIG=0.25; XG=1.2; T=15
AR=0.7; INN=np.sqrt(1-AR**2)   # z stationary std 1, range ~[-3,3] (cos/sin injective)

FREQ=[0.0]
def zdim(tangle): return 2 if tangle else 1
def obs_np(x,z,w):
    zc=np.concatenate([np.cos(w*z),np.sin(w*z)],1)*ALPHA if w>0 else ALPHA*z
    b=np.concatenate([XG*x,zc],1); return b+SIG*np.random.randn(*b.shape)
def obs_t(x,z,w):
    zc=torch.cat([torch.cos(w*z),torch.sin(w*z)],1)*ALPHA if w>0 else ALPHA*z
    return torch.cat([XG*x,zc],1)+SIG*torch.randn(x.shape[0],1+(2 if w>0 else 1))

def pretrain(w,n_ep=2000,steps=1500):
    sx=np.random.randn(n_ep,1); sz=np.random.randn(n_ep,1); Ot=[];Otp=[];A=[];Z=[]
    for t in range(40):
        ax=np.random.randn(n_ep,1); Ot.append(obs_np(sx,sz,w)); Z.append(sz.copy())
        nx=0.9*sx+ax; nz=AR*sz+INN*np.random.randn(n_ep,1); Otp.append(obs_np(nx,nz,w)); A.append(ax); sx,sz=nx,nz
    f=lambda v: torch.tensor(np.concatenate(v),dtype=torch.float32); O,Op,Aa,Zt=f(Ot),f(Otp),f(A),f(Z)
    din=O.shape[1]; enc=torch.nn.Sequential(torch.nn.Linear(din,64),torch.nn.ReLU(),torch.nn.Linear(64,4))
    head=torch.nn.Sequential(torch.nn.Linear(5,64),torch.nn.ReLU(),torch.nn.Linear(64,din)); opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),1e-3)
    for it in range(steps):
        i=torch.randint(0,O.shape[0],(512,)); l=((head(torch.cat([enc(O[i]),Aa[i]],1))-Op[i])**2).mean(); opt.zero_grad();l.backward();opt.step()
    for p in enc.parameters(): p.requires_grad_(False)
    L=enc(O).detach().numpy(); z=Zt.numpy().ravel(); n=int(0.7*len(z))
    def dec(k):
        if k=='lin': Xa=np.c_[L[:n],np.ones(n)]; w,*_=np.linalg.lstsq(Xa,z[:n],rcond=None); p=np.c_[L[n:],np.ones(len(z)-n)]@w
        else:
            m=torch.nn.Sequential(torch.nn.Linear(4,64),torch.nn.ReLU(),torch.nn.Linear(64,1));o=torch.optim.Adam(m.parameters(),1e-2)
            for _ in range(500): o.zero_grad();((m(torch.tensor(L[:n]))-torch.tensor(z[:n,None],dtype=torch.float32))**2).mean().backward();o.step()
            p=m(torch.tensor(L[n:])).detach().numpy().ravel()
        yt=z[n:]; return max(0.,1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum())
    return enc,dec('lin'),dec('mlp')

def mkpol(d,cap):
    if cap=='lin': m=torch.nn.Linear(d,1)
    else: m=torch.nn.Sequential(torch.nn.Linear(d,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
    last=m if cap=='lin' else m[-1]
    with torch.no_grad(): last.weight*=0.05; last.bias*=0.
    return m

def runpol(mode,cap,enc=None,w=0.0,train=True,steps=1000,B=512):
    din={'oracle':2,'blind':1,'latent':4}[mode]; pol=mkpol(din,cap); opt=torch.optim.Adam(pol.parameters(),2e-3)
    def roll(B):
        z=torch.randn(B,1); zs=[]
        for t in range(T): zs.append(z); z=AR*z+INN*torch.randn(B,1)
        x=torch.randn(B,1); err=0.
        for t in range(T):
            zt=zs[t]
            inp=torch.cat([x,zt],1) if mode=='oracle' else (x if mode=='blind' else enc(obs_t(x,zt,w)))
            a=pol(inp); err=err+((x-zt)**2).mean(); x=torch.clamp(0.9*x+a,-15,15)
        return err/T
    if train:
        for it in range(steps):
            l=roll(B); opt.zero_grad();l.backward(); torch.nn.utils.clip_grad_norm_(pol.parameters(),1.0); opt.step()
    with torch.no_grad(): return float(np.mean([roll(4000).item() for _ in range(3)]))

t0=time.time()
print("tangle w | decode_lin decode_mlp | usable_lin usable_mlp")
for w in [0.0,0.7,1.5,2.5]:
    enc,dl,dm=pretrain(w)
    us={}
    for cap in ['lin','mlp']:
        eo=runpol('oracle',cap); eb=runpol('blind',cap); el=runpol('latent',cap,enc,w)
        us[cap]=(eb-el)/(eb-eo+1e-9)
    print(f"  w={w:<4} |   {dl:.2f}      {dm:.2f}    |   {us['lin']:.2f}      {us['mlp']:.2f}")
print(f"\n{time.time()-t0:.0f}s")
