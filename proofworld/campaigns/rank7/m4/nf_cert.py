import pickle, itertools, sympy as sp
P_=2**31-1
g=7
zeta=pow(g,(P_-1)//7,P_)
assert zeta!=1 and pow(zeta,7,P_)==1
def c2(n): return n*(n-1)//2
def normth(e,k):
    """theta_3(q^e zeta^k) = coef * q^qe * atom(e0,k0); returns (coef_mod, qe, atom) or None"""
    k%=7
    s=(e-e%3)//3; e0=e%3
    coef=(-1)**(s%2)*pow(zeta,(-s*k)%7,P_); qe=-(3*c2(s+1))-e0*s+3*s
    if e0==2: e0=1; k=(-k)%7
    if e0==0 and k==0: return None
    if e0==0 and k>3:
        coef=coef*(-1)*pow(zeta,k,P_); k=7-k
    return coef%P_, qe, (e0,k)
ATOMS=[(0,1),(0,2),(0,3)]+[(1,k) for k in range(7)]
AI={a:i for i,a in enumerate(ATOMS)}
def prod(args):
    co=1; qe=0; J=[0]*len(ATOMS)
    for (e,k) in args:
        t=normth(e,k)
        if t is None: return None
        co=co*t[0]%P_; qe+=t[1]; J[AI[t[2]]]+=1
    return co,qe,tuple(J)
def inst(x,y,u,v):   # points as (val, zeta-exp)
    add=lambda a,b:(a[0]+b[0],a[1]+b[1]); sub=lambda a,b:(a[0]-b[0],a[1]-b[1])
    A=prod([add(x,y),sub(x,y),add(u,v),sub(u,v)]); B=prod([add(x,v),sub(x,v),add(u,y),sub(u,y)])
    C=prod([add(y,v),sub(y,v),add(x,u),sub(x,u)])
    uy=sub(u,y)
    poly={}
    for t,s,qs,zs in ((A,1,0,0),(B,-1,0,0),(C,-1,uy[0],uy[1])):
        if t is None: continue
        key=(t[2],t[1]+qs); poly[key]=(poly.get(key,0)+s*t[0]*pow(zeta,zs%7,P_))%P_
    return {k:v for k,v in poly.items() if v}
def normalize(p):
    if len(p)<2: return None
    mq=min(k[1] for k in p); q={(k[0],k[1]-mq):v for k,v in p.items()}
    ks=sorted(q); inv=pow(q[ks[0]],P_-2,P_)
    return tuple(sorted((k,v*inv%P_) for k,v in q.items()))
GEN={}
pts=[(a,b) for a in range(-2,3) for b in range(7)]
for x,y in itertools.product(pts,repeat=2):
    for c,dd in itertools.product(range(7),repeat=2):
        u=(0,c); v=(0,dd)
        if c==dd: continue
        n=normalize(inst(x,y,u,v))
        if n and n not in GEN: GEN[n]=(x,y,u,v)
print("instances",len(GEN))
# target
num,den,TH=pickle.load(open('nf_rem.pkl','rb'))
w,qq,kap=sp.symbols('w q kappa')
tgt={}
for t in sp.Add.make_args(sp.expand(num/kap)):
    c,m=t.as_coeff_Mul(); d=m.as_powers_dict()
    J=[0]*len(ATOMS)
    for (e,k),s in TH.items(): J[AI[(e,k)]]=int(d.get(s,0))
    key=(tuple(J),int(d.get(qq,0)))
    tgt[key]=(tgt.get(key,0)+int(c)*pow(zeta,int(d.get(w,0))%7,P_))%P_
print("target",tgt)
GL=[dict(gk) for gk in GEN]
def div(a,b): return (tuple(x-y for x,y in zip(a[0],b[0])),a[1]-b[1])
def mul(a,b): return (tuple(x+y for x,y in zip(a[0],b[0])),a[1]+b[1])
cands=set()
for i,W in enumerate(GL):
    for tau in W:
        for sig in tgt: cands.add((i,div(sig,tau)))
cands=list(cands)
piv={}
def reduce(rw,comb):
    rw=dict(rw); comb=dict(comb)
    while True:
        cols=[c for c in rw if c in piv]
        if not cols: return rw,comb
        c=cols[0]; f=rw[c]; prow,pcomb=piv[c]
        for k,v in prow.items():
            nv=(rw.get(k,0)-f*v)%P_
            if nv: rw[k]=nv
            else: rw.pop(k,None)
        for k,v in pcomb.items():
            nv=(comb.get(k,0)-f*v)%P_
            if nv: comb[k]=nv
            else: comb.pop(k,None)
for idx,(i,m) in enumerate(cands):
    rw={mul(k,m):v for k,v in GL[i].items()}
    rw,comb=reduce(rw,{idx:1})
    if rw:
        pc=min(rw); inv=pow(rw[pc],P_-2,P_)
        piv[pc]=({k:v*inv%P_ for k,v in rw.items()},{k:v*inv%P_ for k,v in comb.items()})
rest,comb=reduce(tgt,{})
print("found",not rest,"cert size",len(comb))
for idx,v in comb.items():
    i,m=cands[idx]; print(GEN[list(GEN)[i]], m, v)
