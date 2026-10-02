"""7-dissection of E(q) = E(q^49)(x - q y - q^2 + q^5 z), x,y,z series in Q=q^7. Find relations."""
from fractions import Fraction
K = 7*60
def Epow(e, N=K, step=1):
    c = [0]*(N+1); c[0] = 1
    for n in range(1, N//step+1):
        for _ in range(abs(e)):
            if e > 0:
                for i in range(N, n*step-1, -1): c[i] -= c[i-n*step]
            else:
                for i in range(n*step, N+1): c[i] += c[i-n*step]
    return c
def mul(a, b, N):
    c = [0]*(N+1)
    for i, x in enumerate(a[:N+1]):
        if x:
            for j in range(N+1-i): c[i+j] += x*b[j]
    return c
def inv(a, N):
    b = [0]*(N+1); b[0] = Fraction(1, a[0])
    for n in range(1, N+1): b[n] = -sum(a[k]*b[n-k] for k in range(1, n+1))/a[0]
    return b
E = Epow(1)
M = K//7
cls = {r: [E[7*n+r] if 7*n+r <= K else 0 for n in range(M)] for r in range(7)}
print("empty classes:", [r for r in range(7) if not any(cls[r])])
e49 = Epow(1, M-1, 7)   # E(Q^7) as Q-series
ie = inv(e49, M-1)
c2 = cls[2]; print("class2 = -E(Q^7):", [-x for x in c2[:len(e49)]] == e49[:len(c2)])
x = mul(cls[0], ie, M-1); y = [-v for v in mul([cls[1][n] for n in range(M)], ie, M-1)]
# class 5: q^5 z  -> coefficient at 7n+5 is z_n
z = mul(cls[5], ie, M-1)
xyz = mul(mul(x, y, M-1), z, M-1)
print("x y z =", xyz[:8])
print("x:", x[:6], "y:", y[:6], "z:", z[:6])
F = [a - b - c - (8 if n == 1 else 0) for n, (a, b, c) in enumerate(zip(mul(mul(mul(x, x, M-1), x, M-1), y, M-1),
     [0] + [-v for v in mul(mul(mul(y, y, M-1), y, M-1), z, M-1)][:M-1],
     [0, 0] + mul(mul(mul(z, z, M-1), z, M-1), x, M-1)[:M-2]))]
E7 = Epow(1, M-1); E49 = Epow(1, M-1, 7)
r4 = mul(mul(mul(E7, E7, M-1), mul(E7, E7, M-1), M-1), inv(mul(mul(E49, E49, M-1), mul(E49, E49, M-1), M-1), M-1), M-1)
print("F = (E(Q)/E(Q^7))^4:", F[:M-2] == r4[:M-2])
P_ = Epow(-1)
lhs = [P_[7*n+5] for n in range(M-2)]
rhs = [7*a + 49*b for a, b in zip(mul(Epow(3, M-1, 7), Epow(-4, M-1), M-1), [0] + mul(Epow(7, M-1, 7), Epow(-8, M-1), M-1))]
print("Ramanujan mod-7 identity:", lhs == rhs[:M-2], lhs[:5])
