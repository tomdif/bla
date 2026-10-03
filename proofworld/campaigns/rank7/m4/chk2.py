from num import *
def Uzw(z,w,p,R=30): return sum((-1)**(m%2)*p**(m*(3*m+1)//2)*w**m/(1-z*p**m) for m in range(-R,R+1))
def m_(x,p,z): return A(x,z,p)/th(z,p)
p=0.3*cmath.exp(0.2j); z=0.7*cmath.exp(1j); w=0.9*cmath.exp(-0.4j); x=1.3*cmath.exp(0.5j)
P=p**3
print("Uzw split", abs(Uzw(z,w,p)-sum(z**i*A(z**3*p**(1-i)/w,p**(2+i)*w,P) for i in range(3))))
print("m(qx)", abs(m_(P*x,P,z)-(1-x*m_(x,P,z))))
print("inv", abs(m_(x,P,z)-m_(1/x,P,1/z)/x))
print("zshift", abs(m_(x,P,P*z)-m_(x,P,z)))
z0=0.5*cmath.exp(2j)
d=z0*E(P)**3*th(z/z0,P)*th(x*z0*z,P)/(th(z0,P)*th(z,P)*th(x*z0,P)*th(x*z,P))
print("coz", abs(m_(x,P,z)-m_(x,P,z0)-d))
print("A(x,1)", abs(A(x,1,P)+E(P)**3/th(x,P)))
print("A(x,Pz)", abs(A(x,P*z,P)+A(x,z,P)/z))
