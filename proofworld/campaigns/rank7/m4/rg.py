import re
from num import *
from dissect import *
def J(a,b,Q):  # J_{a,b} = theta(Q^b;Q^a) product (Q^b;Q^a)(Q^{a-b};Q^a)(Q^a;Q^a)
    return th(Q**b,Q**a)
FIT={}
for line in open('../R7_dissection_fit.txt'):
    mm=re.match(r'R(\d) ζ\^(\d): (FIT|ZERO) ?(.*)',line.strip())
    if not mm: continue
    c,k,kind,rest=int(mm[1]),int(mm[2]),mm[3],mm[4]
    FIT[(c,k)]=eval(rest) if kind=='FIT' else {}
def basis_val(name,Q):
    sh,f=name.split('*')
    s=int(sh[2:]) if sh!='q^0' else 0
    s=int(sh.split('^')[1])
    if f.startswith('Phi'): v=Phi(int(f[3:]),Q)
    else:
        e=list(map(int,re.findall(r'\^(-?\d+)',f)))
        v=E(Q**7)**e[0]*J(7,1,Q)**e[1]*J(7,2,Q)**e[2]*J(7,3,Q)**e[3]
    return Q**s*(v-(1 if s<0 else 0))
def Rg(c,Q):
    return sum(Z**k*sum(co*basis_val(n,Q) for n,co in FIT[(c,k)].items()) for k in range(6))
if __name__=="__main__":
    q=0.9*cmath.exp(0.05j); Q=q**7
    tot=sum(q**c*Rg(c,Q) for c in range(7))
    print("Rg vs R", abs(tot-Rk(Z,q,N=200)), abs(tot))
