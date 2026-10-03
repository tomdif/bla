import sympy as sp, cmath, math, itertools
from num import th, A, E, Rk, Phi, Z
Qs,ws,EPs=sp.symbols('Q w EP')
Jsym={a:sp.Symbol('J%d'%a) for a in range(1,11)}
Ksym={a:sp.Symbol('K%d'%a) for a in range(0,11)}
def c2(k): return k*(k-1)//2
def thn(sig,a):
    """theta(sig*Q^a; P), P=Q^21 -> sympy expr in J/K symbols"""
    r=a%21; k=(a-r)//21
    # theta(P^k y) = (-1)^k P^{-C(k,2)} y^{-k} theta(y), y = sig Q^r
    pref=(-1)**k*Qs**(-21*c2(k))*(sig**(-k))*Qs**(-r*k)
    if r>10: r=21-r
    if sig==1:
        if r==0: return 0
        return pref*Jsym[r]
    return pref*Ksym[r]
def Delta(X,W,W0):
    """m(X,W)-m(X,W0) with X=Q^X etc (positive sign args); returns expr"""
    return Qs**W0*EPs**3*thn(1,W-W0)*thn(1,X+W0+W)/(thn(1,W0)*thn(1,W)*thn(1,X+W0)*thn(1,X+W))
# canonical orbit reps and chosen W0
W0={}
def red(chi,om):
    """m(Q^chi,P,Q^om) = const + coef*m_o  -> returns (const_expr, coef_expr, o)"""
    const=0; coef=1
    while chi>10:   # m(Q^chi) = 1 - Q^{chi-21} m(Q^{chi-21})
        const+=coef; coef*=-Qs**(chi-21); chi-=21
    while chi<-10:  # m(Y) = (1 - m(PY))/Y
        const+=coef*Qs**(-chi); coef*=-Qs**(-chi); chi+=21
    if chi<0:
        coef*=Qs**(-chi); chi=-chi; om=-om
    w0=W0.setdefault(chi, 2 if chi!=19 else 1)
    if (om-w0)%21!=0:
        const+=coef*Delta(chi,om,w0)
    else:
        k=(om-w0)//21
        assert k==0 or True
    return const,coef,chi

# ---- class-dict q-series: dict c -> expr in Q
def cmul(d1,d2):
    out={}
    for c1,f1 in d1.items():
        for c2_,f2 in d2.items():
            c=c1+c2_; out[c%7]=out.get(c%7,0)+Qs**(c//7)*f1*f2
    return out
def cadd(*ds):
    out={}
    for d in ds:
        for c,f in d.items(): out[c]=out.get(c,0)+f
    return out
def cscale(d,s): return {c:s*f for c,f in d.items()}
def qmono(e,coef=1): return {e%7: coef*Qs**(e//7)}
# multiplier T = theta(zeta^2;q^3)
T={}
for rho in range(7):
    T=cadd(T,qmono(3*c2(rho),(-1)**rho*ws**(2*rho)*thn(1,9+3*rho)))
# ---- S side
mockS={}   # o -> class dict coefficient (already divided by T: i.e. kappa)
CstS={}
for a,pa in ((2,-(1-ws)),(1,-(1-ws)*ws**6)):
    for t in range(7):
        chi=9+a-3*t
        for s in range(7):
            e=3*c2(s)+3*t*(s-1)+a*t
            co=pa*(-1)**s*ws**(2*s+6*t)
            om=3*(3+s+t)
            if om%21==0:
                k=om//21
                val=-(-1)**k*Qs**(-21*c2(k))*EPs**3/thn(1,chi)
                CstS=cadd(CstS,qmono(e,co*val))
            else:
                const,coef,o=red(chi,om)
                CstS=cadd(CstS,qmono(e,co*thn(1,om)*const))
        # mock coefficient: coef_t * mono_t, mono_t = (-1)^t zeta^{4t} q^{a t - 3 C(t+1,2)}
        const,coef,o=red(chi,0)  # coef independent of om
        mockS[o]=cadd(mockS.get(o,{}),qmono(a*t-3*c2(t+1),pa*coef*(-1)**t*ws**(4*t)))
# ---- R_g side
sfr=lambda: None
J7b={b:Jsym[7]*Jsym[b]*Jsym[b+7 if b+7<=10 else 21-b-7]*Jsym[7-b]/EPs**3 for b in (1,2,3)}
E7=Jsym[7]
mockG={}; thetaG={}
def addPhi(k,cls,scale,shift):  # adds scale*Q^shift*(Phi_k - [shift<0]) to class cls
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
# ---- numerics
Qv=0.55*cmath.exp(0.2j); Pv=Qv**21
def ev(expr,wv=Z):
    sub={Qs:Qv,ws:wv,EPs:E(Pv)}
    for a in range(1,11): sub[Jsym[a]]=th(Qv**a,Pv)
    for a in range(0,11): sub[Ksym[a]]=th(-Qv**a,Pv)
    return complex(sp.sympify(expr).evalf(subs=sub)) if not isinstance(expr,(int,float,complex)) else expr
def zred(expr):
    p=sp.Poly(sp.expand(expr),ws) if expr!=0 else None
    return expr
if __name__=="__main__":
    print("orbits S",sorted(mockS),"G",sorted(mockG))
    for o in sorted(set(mockS)|set(mockG)):
        d=cadd(mockS.get(o,{}),cscale(mockG.get(o,{}),-1))
        print("mock",o,{c:abs(ev(f)) for c,f in d.items()})
    # core: T*thetaG == (1-w) T + CstS
    lhs=cmul(T,thetaG); rhs=cadd(cscale(T,1-ws),CstS)
    for c in range(7):
        print("core class",c,abs(ev(lhs.get(c,0))-ev(rhs.get(c,0))), abs(ev(lhs.get(c,0))))
