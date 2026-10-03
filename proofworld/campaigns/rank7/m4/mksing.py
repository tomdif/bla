import pickle, sympy as sp
from core import Qs,Jsym
inst=pickle.load(open('winst.pkl','rb'))
def p2s(items):
    out=[]
    for (J,qe),c in items:
        mon="*".join(f"J{a+1}^{e}" for a,e in enumerate(J) if e)
        out.append(f"({c})*Q^({qe})*{mon}" if qe>=0 else f"({c})/Q^({-qe})*{mon}")
    return "+".join(out)
gens=[p2s(p) for p in inst]
J=Jsym
cubic=J[10]*J[2]*J[3]*Qs**2-J[10]*J[4]*J[9]-J[3]*J[4]*J[5]*Qs**2+J[6]*J[7]**2
s="ring r=(0,Q),(J1,J2,J3,J4,J5,J6,J7,J8,J9,J10),dp;\noption(redSB);\n"
s+="ideal I="+",\n".join(gens)+";\n"
s+="ideal G=std(I);\nprint(size(G));\n"
s+="poly c="+str(sp.expand(cubic)).replace("**","^")+";\n"
s+="print(reduce(J7*J1*c,G));\n"
s+="print(reduce(c,G));\n"
open('t1.sing','w').write(s)
