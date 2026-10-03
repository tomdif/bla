import pickle, sympy as sp
from core import *
from core2 import zcoords
lhs=cmul(T,thetaG); rhs=cadd(cscale(T,1-ws),CstS)
raw={}
for c in range(7):
    d=sp.expand(lhs.get(c,0)-rhs.get(c,0))
    cs=[sp.expand(x) for x in zcoords(d)]
    raw[c]=cs
    print("class",c,[len(sp.Add.make_args(x)) if x!=0 else 0 for x in cs])
pickle.dump(raw,open('raw.pkl','wb'))
print(raw[1][1])
