from fractions import Fraction
K = 60
def mul(a, b):
    c = [0]*(K+1)
    for i, x in enumerate(a):
        if x:
            for j in range(K+1-i): c[i+j] += x*b[j]
    return c
def inv_qfac(m):   # 1/(q)_m
    r = [1]+[0]*K
    for i in range(1, m+1):
        g = [0]*(K+1)
        for t in range(0, K//i+1): g[t*i] = 1
        r = mul(r, g)
    return r
E = [0]*(K+1)
for s in range(-10, 11):
    e = s*(3*s-1)//2
    if 0 <= e <= K: E[e] += (-1)**(s % 2)
def H(a):
    h = [0]*(K+1); b = a*(a+1)//2; sg = (-1)**(a % 2)
    for r in range(0, 20):
        e = b+3*r*r+3*r*a+r
        if e <= K: h[e] += sg
        e = b+3*r*r+3*r*a-r-a
        if r >= 1 and e <= K: h[e] -= sg
    return h
for a in range(0, 6):
    S = [0]*(K+1)
    for n in range(0, 9):
        for l in range(0, 12):
            e = n*n + (l+a)*(l+a-1)//2 + l*(l-1)//2 + (2*l+a)*(n+1)
            if e > K: continue
            t = mul([0]*e + [1] + [0]*(K-e), mul(inv_qfac(l+a), inv_qfac(l)))
            S = [x + (-1)**(a % 2)*y for x, y in zip(S, t)]
    print(a, mul(E, S) == H(a))
def qpoch_inv(lo, m):   # 1/(q^lo;q)_m
    r = [1]+[0]*K
    for i in range(lo, lo+m):
        if i > K: continue
        g = [0]*(K+1)
        for t in range(0, K//i+1): g[t*i] = 1
        r = mul(r, g)
    return r
def mono(e, c=1):
    v = [0]*(K+1)
    if 0 <= e <= K: v[e] = c
    return v
def add(*xs):
    return [sum(t) for t in zip(*xs)]
for k in range(0, 4):
    ok1 = ok2 = True
    for n in range(0, 7):
        bstar = add(*[mul(mono(l), mul(inv_qfac(l), qpoch_inv(k+1, l))) for l in range(n+1)])
        garv = add(*[mul(mono(k*j + j*(j+1)//2, (-1)**j), mul(inv_qfac(n-j), qpoch_inv(k+1, n))) for j in range(n+1)])
        if bstar != garv: ok1 = False
        # Bailey relation for (alpha*, beta*) relative to a=q^k
        rhs = [0]*(K+1)
        for r in range(n+1):
            al = mono(2*r*r+2*k*r+r) if r == 0 else add(mono(2*r*r+2*k*r+r), mono(2*r*r+2*k*r-r-k, -1))
            rhs = add(rhs, mul(al, mul(inv_qfac(n-r), qpoch_inv(k+1, n+r))))
        if rhs != bstar: ok2 = False
    print(f"k={k}: beta* == Garvan Prop2.5 beta': {ok1};  (alpha*,beta*) Bailey pair rel. a=q^k: {ok2}")
