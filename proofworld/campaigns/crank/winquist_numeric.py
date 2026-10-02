"""Check: J(a)J(b)J(ab)J(a/b) = E^2 * W(a,b), W = sum over good (N,M) of eps a^N b^M q^e; then class-6 mod 11 at a=z^5,b=z^2."""
import sympy as sp
from collections import defaultdict
K = 60
T = lambda n: n*(n-1)//2
P = lambda u: u*(3*u-1)//2
R = range(-14, 16)
def cd(A):
    if A % 6 == 1: return (A-1)//6
    if A % 6 == 5: return (A+1)//6
    return None
def W_terms():
    out = {}
    for N in range(-25, 26):
        for M in range(-25, 26):
            A, B = 2*N+2*M-1, 2*N-2*M+1
            c1, c2 = cd(A), cd(B)
            if c1 is None or c2 is None: continue
            e2 = N*N+M*M-N-M + 3*c1*c1 - A*c1 + 3*c2*c2 - B*c2
            assert e2 % 2 == 0
            e = e2//2
            if e <= K: out[(N, M)] = ((-1)**((N+M+c1+c2) % 2), e)
    return out
# symbolic check with generic a,b as dict poly: key (N,M,k)
L = defaultdict(int)
for n1 in R:
  for n2 in R:
    for n3 in R:
      for n4 in R:
        k = T(n1)+T(n2)+T(n3)+T(n4)
        if k <= K: L[(n1+n3+n4, n2+n3-n4, k)] += (-1)**((n1+n2+n3+n4) % 2)
E2 = defaultdict(int)
for u in R:
  for v in R:
    k = P(u)+P(v)
    if k <= K: E2[k] += (-1)**((u+v) % 2)
Wt = W_terms()
Rr = defaultdict(int)
for (N, M), (s, e) in Wt.items():
    for j, c in E2.items():
        if e + j <= K: Rr[(N, M, e+j)] += s*c
bad = [k for k in set(L) | set(Rr) if L[k] != Rr[k]]
print("identity holds to q^%d:" % K, not bad, bad[:5])
# class 6 mod 11 at a=z^5, b=z^2
cls = defaultdict(lambda: defaultdict(int))
for (N, M), (s, e) in Wt.items():
    if e % 11 == 6: cls[e][(5*N+2*M) % 11] += s
print("class-6 coefficients (per zeta power):", {e: dict((k, v) for k, v in d.items() if v) for e, d in cls.items()})
def eps(N, M):
    A, B = 2*N+2*M-1, 2*N-2*M+1
    c1, c2 = cd(A), cd(B)
    if c1 is None or c2 is None: return 0
    return (-1)**((N+M+c1+c2) % 2)
ok1 = all(eps(3-N, M) == -eps(N, M) for N in range(-30, 30) for M in range(-30, 30))
ok2 = all(eps(N, 1-M) == -eps(N, M) for N in range(-30, 30) for M in range(-30, 30))
ok3 = all(eps(3-N, 1-M) == eps(N, M) for N in range(-30, 30) for M in range(-30, 30))
print("eps(3-N,M)=-eps:", ok1, " eps(N,1-M)=-eps:", ok2, " eps(3-N,1-M)=eps:", ok3)
print("e formula:", all(Wt[k][1] == (k[0]**2+k[1]**2-3*k[0]-k[1])//6 for k in Wt))
for r in [(0,0),(0,1),(1,2),(2,2)]:
    print(r, sorted({(eps(N,M), (N//3) % 2, (M//3) % 2) for N in range(r[0], 60, 3) for M in range(r[1], 60, 3)}))
