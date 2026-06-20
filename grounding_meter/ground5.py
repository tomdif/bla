import torch, numpy as np, time
torch.manual_seed(3); np.random.seed(3)
# z = invisible, uncontrolled per-episode goal (both SSL arms = 0). Return arm = REWARD prediction.
def gen(n_ep,T=40,sigma=0.3,xg=1.2,sparsity=1):
    z=np.random.randn(n_ep,1); sx=np.random.randn(n_ep,1); X=[];R=[]
    for t in range(T):
        X.append(sx.copy()); r=-((sx-z)**2)
        if sparsity>1: r=np.where(t%sparsity==0,r,0.0)
        R.append(r); sx=0.9*sx+np.random.randn(n_ep,1)
    return np.stack(X,1),np.stack(R,1),z
def ret_arm(n_ep,sparsity,w=12,steps=2000):
    X,R,z=gen(n_ep,sparsity=sparsity); ne,T,_=X.shape
    F=[];Xt=[];Rt=[];Zs=[]
    for t in range(w,T):
        F.append(np.concatenate([X[:,t-w:t,0],R[:,t-w:t,0]],1)); Xt.append(X[:,t,0]); Rt.append(R[:,t,0] if sparsity==1 else -((X[:,t,0]-z[:,0])**2)); Zs.append(z[:,0])
    F=torch.tensor(np.concatenate(F),dtype=torch.float32); Xc=torch.tensor(np.concatenate(Xt),dtype=torch.float32)[:,None]
    Rt=torch.tensor(np.concatenate(Rt),dtype=torch.float32); Zs=np.concatenate(Zs)
    enc=torch.nn.Sequential(torch.nn.Linear(F.shape[1],128),torch.nn.ReLU(),torch.nn.Linear(128,3))
    head=torch.nn.Sequential(torch.nn.Linear(3+1,64),torch.nn.ReLU(),torch.nn.Linear(64,1))
    opt=torch.optim.Adam(list(enc.parameters())+list(head.parameters()),1e-3); N=F.shape[0]
    for it in range(steps):
        idx=torch.randint(0,N,(512,)); pr=head(torch.cat([enc(F[idx]),Xc[idx]],1))[:,0]; loss=((pr-Rt[idx])**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    with torch.no_grad(): L=enc(F).numpy()
    ntr=int(0.7*len(Zs)); A=np.concatenate([L[:ntr],np.ones((ntr,1))],1); wv,*_=np.linalg.lstsq(A,Zs[:ntr],rcond=None)
    p=np.concatenate([L[ntr:],np.ones((len(Zs)-ntr,1))],1)@wv; yt=Zs[ntr:]
    return max(0.,1-((yt-p)**2).sum()/((yt-yt.mean())**2).sum())
t0=time.time()
print("RETURN arm = reward-prediction; z invisible+uncontrolled (SSL arms=0). decodeZ:")
for lab,sp in [("DENSE (every step)",1),("MID (every 4)",4),("SPARSE (every 12)",12)]:
    r=[f"{ret_arm(ne,sp):.2f}" for ne in [300,1500,6000]]
    print(f"  {lab:<18}: 300ep={r[0]}  1500ep={r[1]}  6000ep={r[2]}")
print(f"{time.time()-t0:.0f}s")
