from num import *
def m_(x,P,z): return A(x,z,P)/th(z,P)
Q=0.6*cmath.exp(0.15j); p=Q**7; P=Q**21
# 1. Phi_k m-form
for k in (1,2,3):
    f=m_(Q**(3*k+7),P,Q**14)-Q**(2*k-7)*m_(Q**(3*k-7),P,Q**28)+Q**k*E(P)**3/(th(Q**7,P)*th(Q**(3*k),P))
    print("Phi",k,abs(f-Phi(k,Q)))
# 2. new formula at zeta
q=0.85*cmath.exp(0.05j)
x=Z
newf=(1-x)*(1-m_(q*q/x**3,q**3,x*x)-m_(q/x**3,q**3,x*x)/x)
print("newformula",abs(newf-Rk(x,q,N=200)))
# 3. split of A(x,z;q^3) n=7 : A = sum_{s,t} coef * A(X,W;q^147)
def split(x,z,qq,n=7):
    Pn=qq**(n*n); tot=0; pieces=[]
    for s in range(n):
        for t in range(n):
            W=(-1)**(n+1)*qq**(n*(n+2*s-1)//2+t*n)*z**n
            X=Pn*qq**(n*(s-1))*x**n*z**n/W
            co=(-1)**s*qq**(s*(s-1)//2+t*(s-1))*z**(s+t)*x**t
            pieces.append((s,t,co,X,W)); tot+=co*A(X,W,Pn)
    return tot,pieces
for xx in (q*q*Z**4,q*Z**4):
    tot,_=split(xx,Z**2,q**3)
    print("split",abs(tot-A(xx,Z**2,q**3)))
