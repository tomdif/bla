import pickle, sympy as sp, random
from core import Qs, Jsym
P_=2**31-1
inst=pickle.load(open('winst.pkl','rb'))
Q0=random.randint(2,P_-2)
def spec(poly_items):
    v={}
    for (J,qe),c in poly_items:
        v[J]=(v.get(J,0)+c*pow(Q0,qe%(P_-1),P_))%P_
    return v
def sym2poly(expr):
    expr=sp.expand(expr); out=[]
    for term in sp.Add.make_args(expr):
        c,m=term.as_coeff_Mul()
        d=m.as_powers_dict()
        J=tuple(int(d.get(Jsym[a],0)) for a in range(1,11)); qe=int(d.get(Qs,0))
        out.append(((J,qe),int(c)))
    return out
def rank_mod(rows,cols):
    M=[[r.get(c,0) for c in cols] for r in rows]
    rk=0; ncol=len(cols)
    for ci in range(ncol):
        piv=None
        for ri in range(rk,len(M)):
            if M[ri][ci]%P_: piv=ri;break
        if piv is None: continue
        M[rk],M[piv]=M[piv],M[rk]
        inv=pow(M[rk][ci],P_-2,P_)
        M[rk]=[x*inv%P_ for x in M[rk]]
        for ri in range(len(M)):
            if ri!=rk and M[ri][ci]%P_:
                f=M[ri][ci]; M[ri]=[(a-f*b)%P_ for a,b in zip(M[ri],M[rk])]
        rk+=1
    return rk
def in_span(target_expr, deg_mult_monos):
    """test target*mono in span of inst*monos (monos: list of J-exponent tuples for multipliers of instances)"""
    pass
J=Jsym
cubic=J[10]*J[2]*J[3]*Qs**2-J[10]*J[4]*J[9]-J[3]*J[4]*J[5]*Qs**2+J[6]*J[7]**2
base=[spec(p) for p in inst]
deg4=[b for b in base]
cols=sorted({k for r in deg4 for k in r})
r0=rank_mod(deg4,cols)
print("instances rank",r0,"cols",len(cols))
for c in range(1,11):
    t=spec(sym2poly(J[c]*cubic))
    cols2=sorted(set(cols)|set(t))
    r1=rank_mod(deg4+[t],cols2)
    print("J%d*cubic in span:"%c, r1==r0)
