import cmath, math
Z=cmath.exp(2j*math.pi/7)
def Rof(p): return max(4,min(60,int(math.sqrt(2*80/max(1e-9,-math.log(abs(p)))))+3))
def th(y,p,R=None):
    R=R or Rof(p)
    return sum((-1)**(k%2)*p**(k*(k-1)//2)*y**k for k in range(-R,R+1))
def A(x,w,p,R=None):
    R=R or Rof(p)
    return sum((-1)**(r%2)*p**(r*(r-1)//2)*w**r/(1-p**(r-1)*x*w) for r in range(-R,R+1))
def E(q,N=400):
    s=1
    for n in range(1,N): s*=1-q**n
    return s
def Rk(z,q,N=80):
    s=0;t=1
    for n in range(N):
        if n>0: t/= (1-z*q**n)*(1-q**n/z)
        s+=q**(n*n)*t
    return s
def Phi(k,Q,N=60):  # sum Q^{7n^2}/((Q^k;Q^7)_{n+1}(Q^{7-k};Q^7)_n)
    s=0;a=1/(1-Q**k);b=1
    for n in range(N):
        s+=Q**(7*n*n)*a*b
        a/=1-Q**(k+7*(n+1)); b/=1-Q**(7-k+7*n)
    return s
if __name__=="__main__":
    q=0.23*cmath.exp(0.3j); z=Z
    lhs=E(q)*Rk(z,q)
    rhs=(1-z)*(-q*A(z**3*q,1/q,q**3)-z*A(z**3,1,q**3)-z*z/q*A(z**3/q,q,q**3))
    print("bridge",abs(lhs-rhs))
    # Phi_k AL form guess: E(Q^7) Phi_k = sum_j Q^{kj} A(Q^{3k+7(1-j)}, Q^{7(2+j)}; Q^21)
    Q=0.5*cmath.exp(0.2j)
    for k in (1,2,3):
        g=E(Q**7)*Phi(k,Q)
        u=sum((-1)**(n%2)*Q**(7*n*(3*n+1)//2)/(1-Q**(k+7*n)) for n in range(-40,41))
        print(k,"U form",abs(g-u))
