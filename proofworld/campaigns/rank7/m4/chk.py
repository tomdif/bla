from rg import *
q=0.85*cmath.exp(0.05j); Q=q**7
w=Z
for c in range(7):
    v=sum(w**(-j*c)*Rk(Z,w**j*q,N=200) for j in range(7))/7/q**c
    print(c, abs(v-Rg(c,Q)), v, Rg(c,Q))
