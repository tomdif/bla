import pickle, sympy as sp
import corelean as CL
from corelean import red, ev, evd, T, mockS, constS, corrS, Qs, ws, EPs, Jsym, thn, cmul, cadd, cscale, qmono
mockG={}; thetaG={}
J7b={b:Jsym[7]*Jsym[b]*Jsym[b+7 if b+7<=10 else 21-b-7]*Jsym[7-b]/EPs**3 for b in (1,2,3)}
E7=Jsym[7]
def addPhi(k,cls,scale,shift):
    global thetaG
    const=Qs**k*EPs**3/(Jsym[7]*thn(1,3*k))
    for chi,om,cf in ((3*k+7,14,1),(3*k-7,28,-Qs**(2*k-7))):
        c0,cf2,o=red(chi,om)
        const+=cf*c0
        mockG[o]=cadd(mockG.get(o,{}),{cls:scale*Qs**shift*cf*cf2})
    thetaG=cadd(thetaG,{cls:scale*Qs**shift*(const-(1 if shift<0 else 0))})
s1=ws**2+ws**3+ws**4+ws**5
J71,J72,J73=J7b[1],J7b[2],J7b[3]
addPhi(1,0,3+s1,0)
thetaG=cadd(thetaG,{0:-(2+s1)*E7**2*J73/(J71*J72)})
thetaG=cadd(thetaG,{1:E7**2/J71})
addPhi(3,2,-(1+2*ws**2+ws**3+ws**4+2*ws**5),-1)
thetaG=cadd(thetaG,{2:-(1+s1)*E7**2*J72/(J71*J73)})
thetaG=cadd(thetaG,{3:(1+ws**2+ws**5)/Qs*(E7**2*J72*J73/J71**3-E7**2*J73**3/(J71**2*J72**2))})
thetaG=cadd(thetaG,{4:(ws**2+ws**5)/Qs*(-E7**2*J72**2/J71**3+E7**2*J73**2/(J71**2*J72))})
addPhi(2,6,ws**2-ws**3-ws**4+ws**5,-1)
thetaG=cadd(thetaG,{6:(1+ws**3+ws**4)/Qs*(-E7**2*J72/J71**2+E7**2*J73**2/(J71*J72**2))})
for o in sorted(set(mockS)|set(mockG)):
    break
    d=cadd(cscale(mockS.get(o,{}),-(1-ws)),cscale(mockG.get(o,{}),-1))
    print("mock",o,{c:abs(ev(f)) for c,f in d.items()})
lhs=cmul(T,thetaG)
rhs=cadd(cscale(cadd(T,cscale(cmul(T,constS),-1)),1-ws),cscale(corrS,1-ws))
for c in range(7):
    print("class",c,abs(ev(lhs.get(c,0))-ev(rhs.get(c,0))),abs(ev(lhs.get(c,0))))
pickle.dump((lhs,rhs,thetaG),open('corelean2.pkl','wb'))
