"""Planted-truth check of the 5-dissection route to Ramanujan's identity Σ p(5n+4) q^n = 5 E(q^5)^5 / E(q)^6."""
from fractions import Fraction
import sympy as sp
K = 200                                            # q-degree
def Epow(e, N=K, step=1):                          # E(q^step)^e, exact
    c = [0] * (N + 1); c[0] = 1
    for n in range(1, N // step + 1):
        for _ in range(abs(e)):
            if e > 0:
                for i in range(N, n * step - 1, -1): c[i] -= c[i - n * step]
            else:
                for i in range(n * step, N + 1): c[i] += c[i - n * step]
    return c
def mul(a, b, N):
    c = [0] * (N + 1)
    for i, x in enumerate(a[:N + 1]):
        if x:
            for j in range(N + 1 - i): c[i + j] += x * b[j]
    return c
def inv(a, N):                                     # 1/a, a[0] = ±1
    b = [0] * (N + 1); b[0] = Fraction(1, a[0])
    for n in range(1, N + 1): b[n] = -sum(a[k] * b[n - k] for k in range(1, n + 1)) / a[0]
    return b
E = Epow(1)
M = K // 5                                         # Q-degree available (Q = q^5)
dis = lambda f, r: [f[5 * n + r] for n in range(M) if 5 * n + r <= K]
c = {r: dis(E, r) for r in range(5)}
e = Epow(1, M - 1, 5)                              # E(Q^5) as a Q-series
print("(1) class 1 of E = −E(Q^5):", c[1] == [-x for x in e[:len(c[1])]],
      "| classes 3,4 empty:", not any(c[3]) and not any(c[4]))
m = min(len(c[0]), len(c[2])) - 1
alpha = mul(c[0], inv(e, m), m); beta = [-x for x in mul(c[2], inv(e, m), m)]
ab = mul(alpha, beta, m)
print("(2) αβ = 1:", ab[0] == 1 and not any(ab[1:]))
a5, b5 = alpha[:], beta[:]
for _ in range(4): a5 = mul(a5, alpha, m); b5 = mul(b5, beta, m)
Nser = [a5[n] - (11 if n == 1 else 0) - (b5[n - 2] if n >= 2 else 0) for n in range(m + 1)]
rhs = mul(Epow(6, m), inv(Epow(6, m, 5), m), m)
print("(3) α⁵ − 11Q − Q²β⁵ = E(Q)⁶/E(Q⁵)⁶:", Nser == rhs)
P = Epow(-1)                                       # partition numbers
lhs = [P[5 * n + 4] for n in range(m + 1)]
rhs = [5 * x for x in mul(Epow(5, m, 5), inv(Epow(6, m), m), m)]
print("(4) Σ p(5n+4) Qⁿ = 5 E(Q⁵)⁵/E(Q)⁶:", lhs == rhs, "  first:", lhs[:6])
# symbolic: the two polynomial identities (with AB = 1)
A, B, t, w = sp.symbols('A B t w')
Qp = A**4 + t*A**3 + 2*t**2*A**2 + 3*t**3*A + 5*t**4 - 3*t**5*B + 2*t**6*B**2 - t**7*B**3 + t**8*B**4
target = A**5 - 11*t**5 - t**10*B**5
G = sp.groebner([A*B - 1], A, B, t, order='lex')
print("(5) (A − t − t²B)·Q = A⁵ − 11t⁵ − t¹⁰B⁵ mod (AB−1):", G.reduce(sp.expand((A - t - t**2*B)*Qp - target))[1] == 0)
prod = sp.expand(sp.prod([A - w**j*t - w**(2*j)*t**2*B for j in range(5)]))
G2 = sp.groebner([A*B - 1, w**4 + w**3 + w**2 + w + 1], A, B, t, w, order='lex')
print("(6) ∏_j (A − ω^j t − ω^{2j} t² B) = A⁵ − 11t⁵ − t¹⁰B⁵:", G2.reduce(sp.expand(prod - target))[1] == 0)
print("    class-4 terms of Q (t^j, j≡4 mod 5):", [sp.expand(Qp).coeff(t, j) for j in (4, 9)])
