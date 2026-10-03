"""Orbit pieces G_{c,o} of the Hecke–Rogers series at p=7; test which are theta products."""
from collections import defaultdict
p = 7; K = 7 * 120
def orbit(j): return min(j % p, (-j) % p)
def pieces(terms):
    G = defaultdict(lambda: [0] * (K + 1))
    for e, sgn, j in terms:
        if 0 <= e <= K: G[(e % p, orbit(j))][e] += sgn
    return G
t18 = []
for n in range(0, 90):
    for j in range(0, n // 2 + 1):
        if (n*n - 3*j*j + n - j) % 2 == 0:
            e = (n*n - 3*j*j + n - j) // 2; s = (-1) ** (n + j)
            t18.append((e, s, n - 3*j)); t18.append((e, s, 3*j - n))
    for j in range(1, n // 2 + 1):
        if (n*n - 3*j*j + n + j) % 2 == 0:
            e = (n*n - 3*j*j + n + j) // 2; s = (-1) ** (n + j)
            t18.append((e, s, n - 3*j + 1)); t18.append((e, s, 3*j - n - 1))
t19 = []
for n in range(0, 80):
    for j in range(-n, n + 1):
        b = n*(3*n+1)//2 - j*j
        t19.append((b, (-1)**(j % 2), j)); t19.append((b + 2*n + 1, -(-1)**(j % 2), j))
def prod_exps(f, N):
    # f = c q^s (1 + ...) ; return a_n with f/(c q^s) = prod (1-q^n)^{a_n}, n<=N
    s = next(i for i, v in enumerate(f) if v)
    g = [x / f[s] for x in f[s:]]
    from math import log
    a = []
    L = g[:N + 1]
    # log-derivative method: n*b_n where log g = sum b_n q^n
    # compute log via recurrence
    lg = [0.0] * (N + 1)
    for n in range(1, N + 1):
        lg[n] = (n * L[n] - sum(k * lg[k] * L[n - k] for k in range(1, n))) / n if n < len(L) else 0
    # log prod (1-q^m)^{a_m} = -sum_m a_m sum_k q^{mk}/k  => n*lg[n] = -sum_{m|n} m a_m
    for n in range(1, N + 1):
        tot = n * lg[n] + sum(m * a[m - 1] for m in range(1, n) if n % m == 0)
        a.append(-tot / n)
    return s, f[s], a
for name, T in (("2.18", t18), ("2.19", t19)):
    G = pieces(T)
    print("=====", name)
    for (c, o) in sorted(G):
        f = G[(c, o)]
        if not any(f): continue
        sub = [f[c + p * m] for m in range((K - c) // p + 1)]   # as series in Q=q^7
        if not any(sub[:100]): print(c, o, "ZERO"); continue
        s, lead, a = prod_exps(sub, 60)
        ai = [round(x) for x in a]
        integral = all(abs(x - round(x)) < 1e-6 for x in a)
        per = None
        for P in (1, 2, 7, 14, 49, 98):
            if P <= 30 and integral and all(ai[n] == ai[n % P] for n in range(P, 60)): per = P; break
        print(c, o, "lead q^%d·%s" % (s, lead), "integral" if integral else "nonint", "period", per, ai[:16] if integral else "")
