"""Two-variable series of (1+z+..+z^{m-1})(z^m q)(z^{-m} q)(q) R(z;q); print z^c coefficients (sparse?)."""
import sys
from collections import defaultdict
m = int(sys.argv[1]); N = int(sys.argv[2])
# series: dict n -> dict c -> int
def mul(A, B):
    C = defaultdict(lambda: defaultdict(int))
    for n1, d1 in A.items():
        for n2, d2 in B.items():
            if n1 + n2 > N: continue
            for c1, v1 in d1.items():
                for c2, v2 in d2.items(): C[n1 + n2][c1 + c2] += v1 * v2
    return C
def one(): A = defaultdict(lambda: defaultdict(int)); A[0][0] = 1; return A
def factor(c, e):  # (1 - z^c q^e)
    A = one(); A[e][c] -= 1; return A
def geo(c, e):  # 1/(1 - z^c q^e)
    A = defaultdict(lambda: defaultdict(int))
    t = 0
    while e * t <= N: A[e * t][c * t] += 1; t += 1
    return A
# R(z;q) Durfee
R = defaultdict(lambda: defaultdict(int))
cur = one(); k = 0
while k * k <= N:
    for n, d in cur.items():
        if n + k * k <= N:
            for c, v in d.items(): R[n + k * k][c] += v
    k += 1
    cur = mul(mul(cur, geo(1, k)), geo(-1, k))
P = one()
for n in range(1, N + 1):
    P = mul(P, factor(m, n)); P = mul(P, factor(-m, n)); P = mul(P, factor(0, n))
L = defaultdict(lambda: defaultdict(int)); L[0][0] = 0
pre = defaultdict(lambda: defaultdict(int))
for i in range(m): pre[0][i] = 1
H = mul(mul(pre, P), R)
byc = defaultdict(dict)
for n, d in H.items():
    for c, v in d.items():
        if v: byc[c][n] = v
for c in sorted(byc):
    if -6 <= c <= 9:
        terms = sorted(byc[c].items())
        print(c, len(terms), terms[:14])
