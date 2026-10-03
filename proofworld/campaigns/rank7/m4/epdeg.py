import pickle, sympy as sp
from corelean import Qs, ws, EPs
from core2 import zcoords
lhs,rhs,thetaG=pickle.load(open('corelean2.pkl','rb'))
for c in range(7):
    d=sp.expand(lhs.get(c,0)-rhs.get(c,0))
    for i,x in enumerate(zcoords(d)):
        x=sp.expand(x)
        if x==0: continue
        degs={}
        for t in sp.Add.make_args(x):
            dg=sp.degree(t,EPs) if t.has(EPs) else 0
            degs[dg]=degs.get(dg,0)+t
        print(c,i,{k:(sp.simplify(v)==0) for k,v in degs.items()})
