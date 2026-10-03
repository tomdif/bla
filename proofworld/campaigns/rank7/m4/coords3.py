import pickle, sympy as sp, time
from corelean import Qs, ws, EPs, Jsym
from core2 import zcoords
from sparsecert import sym2dict, solve
lhs,rhs,thetaG=pickle.load(open('corelean3.pkl','rb'))
raw={}; res={}
for c in range(7):
    d=sp.expand(lhs.get(c,0)-rhs.get(c,0))
    cs=[sp.expand(x) for x in zcoords(d)]
    raw[c]=cs
    for i,x in enumerate(cs):
        if x==0: continue
        tgt=sym2dict(x)
        t0=time.time(); ok,n,cert=solve(tgt,rounds=1)
        res[(c,i)]=(ok,cert,tgt)
        print(c,i,"terms",len(tgt),"found",ok,"cert",len(cert),"%.1fs"%(time.time()-t0),flush=True)
import os
pickle.dump((raw,res),open(os.environ.get('OUT','coords3.pkl'),'wb'))
print('SCORE',sum(1 for v in res.values() if v[0]),len(res),sum(len(v[1]) for v in res.values()))
