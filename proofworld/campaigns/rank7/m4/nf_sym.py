import sympy as sp
w,qq,kap,X1,X2=sp.symbols('w q kappa X1 X2')
TH={}
def th(e,k):
    """theta_3(q^e w^k) normalized symbolic, base q^3"""
    k%=7
    c=1
    # shift e into {0,1,2}
    s=(e - e%3)//3; e0=e%3
    # theta(e0+3s)(c) = q^{-(3 c2(s+1)) - e0 s + 3 s} (-1)^s c^{-s} theta(e0)(c)
    c2=lambda n:n*(n-1)//2
    pref=qq**(-(3*c2(s+1))-e0*s+3*s)*(-1)**(s%2)*w**((-s*k)%7)
    if e0==2:   # theta(2)(c) = theta(1)(c^{-1})
        e0=1; k=(-k)%7
    if e0==0 and k==0: return 0
    if e0==0 and k>3:   # theta(0)(c^{-1}) = -c^{-1} theta(0)(c)
        pref*=-w**((-(7-k))%7) if False else -w**(k%7); k=7-k
        # check: theta(0)(w^k) with k>3 : w^k = (w^{7-k})^{-1} -> = -(w^{7-k})^{-1} theta(0)(w^{7-k}) = -w^{k} theta(0)(w^{7-k})
    key=(e0,k)
    if key not in TH: TH[key]=sp.Symbol(f"t{e0}_{k}")
    return pref*TH[key]
z=w
T=th(0,2)
E_=th(1,0)
Ai1=kap/th(0,3)
def coz1(b1,k1): return th(b1,k1)*X1/th(1,0)-qq*kap*th(-1+b1,k1)*th(2+b1,3+k1)/(th(1,0)*th(2,3)*th(1+b1,3+k1))
def coz2(b1,k1): return th(b1,k1)*X2/th(2,0)-qq**2*kap*th(-2+b1,k1)*th(1+b1,3+k1)/(th(2,0)*th(1,3)*th(-1+b1,3+k1))
Ab10=coz1(0,5); Ai0=coz1(-1,0)
Abm10=coz2(0,5); Ai2=coz2(1,0)
Ab1m3=-Ab10/(qq**3*w**2)
Abm20=(Ab1m3-th(-3,5))*qq**5/w
B2=-Abm20/(w**2*qq**2)
B1=-w**5/qq*Abm10
lhs=E_*(T-B2-w**6*B1)
rhs=T*(-qq*Ai0-w*Ai1-w**2/qq*Ai2)
d=sp.expand(lhs-rhs)
print("X1 coeff:",sp.simplify(sp.expand(d).coeff(X1)))
print("X2 coeff:",sp.simplify(sp.expand(d).coeff(X2)))
rem=sp.expand(d.subs({X1:0,X2:0}))
remt=sp.together(rem)
num,den=sp.fraction(remt)
num=sp.expand(num)
# reduce w powers mod 7
def wred(e):
    e=sp.expand(e); out=0
    for t in sp.Add.make_args(e):
        pw=sp.degree(t,w) if t.has(w) else 0
        out+=t/w**pw*w**(pw%7)
    return sp.expand(out)
num=wred(num)
print("remainder numerator terms:",len(sp.Add.make_args(num)))
print(sp.factor(num))
print("den",den)
import pickle; pickle.dump((num,den,TH),open('nf_rem.pkl','wb'))
