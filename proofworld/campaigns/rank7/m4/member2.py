from member import *
def mulmono(v,e):  # multiply specialized vector by J_e
    out={}
    for J_,c in v.items():
        J2=list(J_); J2[e-1]+=1; out[tuple(J2)]=c
    return out
deg5=[mulmono(b,e) for b in base for e in range(1,11)]
cols=sorted({k for r in deg5 for k in r})
r0=rank_mod(deg5,cols); print("deg5 rank",r0,len(cols))
import itertools
found=[]
for c,d in itertools.combinations_with_replacement(range(1,11),2):
    t=spec(sym2poly(J[c]*J[d]*cubic))
    cols2=sorted(set(cols)|set(t))
    if rank_mod(deg5+[t],cols2)==r0: found.append((c,d))
print("cubic multipliers found:",found)
q5=-J[10]**2*J[4]*J[5]*J[6] + J[10]*J[2]*J[4]**2*J[6]*Qs**2 + J[10]*J[2]*J[7]*J[8]*J[9] - J[10]*J[4]**2*J[5]*J[6]*Qs + J[4]**3*J[8]*J[9]*Qs
t=spec(sym2poly(q5)); cols2=sorted(set(cols)|set(t))
print("quintic in span:",rank_mod(deg5+[t],cols2)==r0)
