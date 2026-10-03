# Lean-exact decomposition: m_split (n=7, base q^147) of A(q^a z^4, z^2; q^3), z = zeta7
import sympy as sp, cmath, math
from num import th, A, E, Rk, Z
from core import Qs, ws, EPs, Jsym, thn, cmul, cadd, cscale, qmono, c2
import os, json
WO={o:1 for o in (1,2,4,5,7,8,10)}
if os.environ.get('WO'): WO={int(k):v for k,v in json.loads(os.environ['WO']).items()}
W11=int(os.environ.get('W11','1'))
def Delta(X,W,W0):   # m(Q^X,Q^W) - m(Q^X,Q^W0)
    return Qs**W0*EPs**3*thn(1,W-W0)*thn(1,X+W0+W)/(thn(1,W0)*thn(1,W)*thn(1,X+W0)*thn(1,X+W))
def red(chi,om):
    """symbolic: m(Q^chi,P,Q^om) = const + coef*m_o, m_o = m(Q^o,Q^{WO[o]})"""
    const=0; coef=1
    while chi>10:
        const+=coef; coef*=-Qs**(chi-21); chi-=21
    while chi<-10:
        const+=coef*Qs**(-chi); coef*=-Qs**(-chi); chi+=21
    if chi<0:
        coef*=Qs**(-chi); chi=-chi; om=-om
    w0=WO[chi]
    if (om-w0)%21: const+=coef*Delta(chi,om,w0)
    return const,coef,chi
# choose bz (Q-units) per (a,s)
def choose_w(a,s):
    chi=9+a-3*s
    if 1<=chi<=10: return WO[chi]
    if -10<=chi<=-1: return 21-WO[-chi]
    if chi==11: return W11          # orbit 10 via x-shift + coz
    raise ValueError
T={}
for rho in range(7):   # theta_dissect: T = sum_rho mono(3c2(-rho))((-1)^{-rho} zeta^{-2rho}) theta(Q^{9-3rho})
    T=cadd(T,qmono(3*c2(-rho),(-1)**rho*ws**((-2*rho)%7)*thn(1,9-3*rho)))
mockS={}; constS={}; corrS={}
for a,pa in ((2,1),(1,ws**6)):
    for s in range(7):
        chi=9+a-3*s; w=choose_w(a,s)
        assert 1<=w<=20 and 1<=chi+w<=20, (a,s,chi,w)
        e_s=a*s-3*c2(s+1); c_s=(-1)**s*ws**((4*s)%7)
        const,coef,o=red(chi,w)
        mockS[o]=cadd(mockS.get(o,{}),qmono(e_s,pa*c_s*coef))
        constS=cadd(constS,qmono(e_s,pa*c_s*const))
        for rho in range(7):
            spE=3*c2(-rho)+3*s*(-rho-1)+a*s
            spC=(-1)**rho*ws**((-2*rho+6*s)%7)
            spB=9-3*rho+3*s
            corr=Qs**w*(-EPs**3)*thn(1,-w+spB)*thn(1,chi+w+spB)/(thn(1,w)*thn(1,chi+w)*thn(1,chi+spB))
            corrS=cadd(corrS,qmono(spE,pa*spC*corr))
# R = (1-z)[1 - sum pa mono_s m_{a,s}] + (1-z) corr/T   where A_a = sum mono_s T m - corr
# numeric check
q=0.85*cmath.exp(0.05j); Qv=q**7; Pv=Qv**21
def ev(expr):
    sub={Qs:Qv,ws:Z,EPs:E(Pv)}
    for k in range(1,11): sub[Jsym[k]]=th(Qv**k,Pv)
    return complex(sp.sympify(expr).evalf(subs=sub,n=20))
def evd(d): return sum(q**c*ev(f) for c,f in d.items())
def m_o(o): return A(Qv**o,Qv**WO[o],Pv)/th(Qv**WO[o],Pv)
Tv=evd(T)
if 0: print("T check",abs(Tv-th(Z**2,q**3)))
mock=sum(evd(d)*m_o(o) for o,d in mockS.items())
Rnum=(1-Z)*(1-mock-evd(constS))+(1-Z)*evd(corrS)/Tv
print("R check",abs(Rnum-Rk(Z,q,N=250)))
import pickle
pickle.dump((mockS,constS,corrS,T),open('corelean.pkl','wb'))
if 0:
  for o in sorted(mockS): print("orbit",o,{c:sp.simplify(f) for c,f in mockS[o].items()})
