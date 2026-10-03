from core import *
import pickle
def zcoords(expr):
    expr=sp.expand(expr)
    co=[0]*7
    for term in sp.Add.make_args(expr):
        pw=sp.degree(term,ws) if term.has(ws) else 0
        co[pw%7]+=term/ws**pw
    return [sp.together(co[i]-co[6]) for i in range(6)]  # basis 1..w^5
lhs=cmul(T,thetaG); rhs=cadd(cscale(T,1-ws),CstS)
res={}
for c in range(7):
    d=sp.expand(lhs.get(c,0)-rhs.get(c,0))
    cs=zcoords(d)
    res[c]=cs
    for i,x in enumerate(cs):
        num,den=sp.fraction(sp.together(x))
        num=sp.expand(num)
        n=len(sp.Add.make_args(num)) if num!=0 else 0
        print(c,i,"numerator terms",n, "check", abs(ev(x)) if n else 0)
pickle.dump(res,open('core_res.pkl','wb'))
