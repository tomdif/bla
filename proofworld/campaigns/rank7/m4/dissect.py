# Direct 7-dissection of U(zeta;q) and check of class identities E*R_g = (1-zeta) U(zeta)
from num import *
def Uzw(z,w,p,R=25,skip0=False):
    return sum((-1)**(m%2)*p**(m*(3*m+1)//2)*w**m/(1-z*p**m) for m in range(-R,R+1) if not(skip0 and m==0))
def V(rho,j,Q):
    p=Q**7
    return Uzw(Q**rho,Q**(3*rho+j-3),p,skip0=(rho==0))
def classes_U(Q):
    # returns dict c -> value of class-c component U_c(Q) (with zeta coefficients) s.t. U(zeta;q)=sum q^c U_c(q^7)
    out={c:0 for c in range(7)}
    out[0]+=1/(1-Z)
    for rho in range(7):
        for j in range(7):
            e=rho*(3*rho+1)//2+rho*j
            c=e%7
            out[c]+=(-1)**(rho%2)*Z**j*Q**((e-c)//7)*V(rho,j,Q)
    return out
if __name__=="__main__":
    q=0.9*cmath.exp(0.05j); Q=q**7
    cu=classes_U(Q)
    tot=sum(q**c*cu[c] for c in range(7))
    print("dissection check",abs(tot-Uzw(Z,1,q)))
