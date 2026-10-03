import pickle, itertools, sympy as sp
from sparsecert import *
res=pickle.load(open('certs.pkl','rb'))
raw=pickle.load(open('raw.pkl','rb'))
out={}
for c in range(7):
    ids=[i for i in range(6) if raw[c][i]!=0]
    tg={i:sym2dict(raw[c][i]) for i in ids}
    keys=sorted({k for i in ids for k in tg[i]})
    M=sp.Matrix([[tg[i].get(k,0) for k in keys] for i in ids])
    rk=M.rank()
    good=[i for i in ids if res[(c,i)][0]]
    Mg=sp.Matrix([[tg[i].get(k,0) for k in keys] for i in good]) if good else sp.zeros(0,len(keys))
    rg=Mg.rank() if good else 0
    extra=[]
    if rg<rk:
        for i,j in itertools.combinations(ids,2):
            for a,b in itertools.product(range(-2,3),repeat=2):
                if a==0 or b==0: continue
                t={}
                for k in keys:
                    v=a*tg[i].get(k,0)+b*tg[j].get(k,0)
                    if v: t[k]=v
                if not t: continue
                ok,n,cert=solve(t,rounds=1)
                if ok:
                    extra.append(((i,a),(j,b),len(cert)))
                    rows=list(Mg.tolist())+[[t.get(k,0) for k in keys]]
                    if sp.Matrix(rows).rank()>rg:
                        Mg=sp.Matrix(rows); rg+=1
                if rg==rk: break
            if rg==rk: break
    print("class",c,"rank",rk,"good",good,"rank good",rg,"extra",extra[:3],flush=True)
