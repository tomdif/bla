"""(2.20) per z^c:  Σ_m q^{C(m,2)} h1_{|c-m|}(q) = X_c(q),
h1_b = (-1)^b q^{b(b+1)/2} Σ_j q^{j²+bj} Σ_{l≤j} q^l/((q)_l (q)_{l+b}),
X_c = Σ over n∈{c-1, -c}, n≥0, |j|≤n/2 of (-1)^{n+j} q^{(n²-3j²)/2+(n-j)/2}."""
K = 60
def mul(a, b):
    c = [0] * (K + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(K + 1 - i): c[i + j] += x * b[j]
    return c
def inv_qfac(l):
    f = [0] * (K + 1); f[0] = 1
    for i in range(1, l + 1):  # multiply by 1/(1-q^i)
        for n in range(i, K + 1): f[n] += f[n - i]
    return f
IQ = [inv_qfac(l) for l in range(K + 2)]
def h1(b):
    s = [0] * (K + 1)
    for j in range(0, K + 1):
        e0 = b * (b + 1) // 2 + j * j + b * j
        if e0 > K: break
        for l in range(0, j + 1):
            if e0 + l > K: break
            t = mul(IQ[l], IQ[l + b])
            for n in range(K + 1 - e0 - l): s[n + e0 + l] += (-1) ** b * t[n]
    return s
H = {b: h1(b) for b in range(0, 2 * K)}
def lhs(c):
    s = [0] * (K + 1)
    for m in range(-K - 2, K + 3):
        e = m * (m - 1) // 2
        if e > K: continue
        b = abs(c - m)
        if b * (b + 1) // 2 > K: continue
        for n in range(K + 1 - e): s[n + e] += H[b][n]
    return s
def X(c):
    s = [0] * (K + 1)
    for n in (c - 1, -c):
        if n < 0: continue
        for j in range(-(n // 2), n // 2 + 1):
            e = (n * n - 3 * j * j + n - j) // 2
            if 0 <= e <= K: s[e] += (-1) ** (n + j)
    return s
for c in range(-4, 6):
    L, R = lhs(c), X(c)
    print(c, L == R, [x for x in R[:20]])
