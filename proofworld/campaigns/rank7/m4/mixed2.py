import pickle, sympy as sp, time, sys
import corelean2 as C2
from corelean import red, T, mockS, constS, corrS, Qs, ws, EPs, Jsym, thn, cmul, cadd, cscale, qmono
from core2 import zcoords
Ks={b:sp.Symbol('K%d'%b) for b in (1,2,3)}
# rebuild thetaG with K atoms
thetaG={}
def addPhi(k,cls,scale,shift):
    global thetaG
    const=Qs**k*EPs**3/(Jsym[7]*thn(1,3*k))
    for chi,om,cf in ((3*k+7,14,1),(3*k-7,28,-Qs**(2*k-7))):
        c0,cf2,o=red(chi,om); const+=cf*c0
    thetaG=cadd(thetaG,{cls:scale*Qs**shift*(const-(1 if shift<0 else 0))})
s1=ws**2+ws**3+ws**4+ws**5
J71,J72,J73=Ks[1],Ks[2],Ks[3]; E7=Jsym[7]
addPhi(1,0,3+s1,0)
thetaG=cadd(thetaG,{0:-(2+s1)*E7**2*J73/(J71*J72)})
thetaG=cadd(thetaG,{1:E7**2/J71})
addPhi(3,2,-(1+2*ws**2+ws**3+ws**4+2*ws**5),-1)
thetaG=cadd(thetaG,{2:-(1+s1)*E7**2*J72/(J71*J73)})
thetaG=cadd(thetaG,{3:(1+ws**2+ws**5)/Qs*(E7**2*J72*J73/J71**3-E7**2*J73**3/(J71**2*J72**2))})
thetaG=cadd(thetaG,{4:(ws**2+ws**5)/Qs*(-E7**2*J72**2/J71**3+E7**2*J73**2/(J71**2*J72))})
addPhi(2,6,ws**2-ws**3-ws**4+ws**5,-1)
thetaG=cadd(thetaG,{6:(1+ws**3+ws**4)/Qs*(-E7**2*J72/J71**2+E7**2*J73**2/(J71*J72**2))})
lhs=cmul(T,thetaG)
rhs=cadd(cscale(cadd(T,cscale(cmul(T,constS),-1)),1-ws),cscale(corrS,1-ws))
SYMS=[Jsym[a] for a in range(1,11)]+[Ks[1],Ks[2],Ks[3],EPs]
def tod(expr):
    expr=sp.expand(sp.nsimplify(expr)); out={}
    for term in sp.Add.make_args(expr):
        c,m=term.as_coeff_Mul(); d=m.as_powers_dict()
        key=(tuple(int(d.get(s,0)) for s in SYMS),int(d.get(Qs,0)))
        out[key]=out.get(key,0)+sp.Rational(c)
    return {k:v for k,v in out.items() if v}
targets={}
for c in range(7):
    d=sp.expand(lhs.get(c,0)-rhs.get(c,0))
    for i,x in enumerate(zcoords(d)):
        x=sp.expand(x)
        if x!=0: targets[(c,i)]=tod(x)
pickle.dump((targets,thetaG),open('mixed_targets.pkl','wb'))
print({k:len(v) for k,v in targets.items()})
