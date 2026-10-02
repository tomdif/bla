from fractions import Fraction
K = 400
def mul(a, b, N=K):
    c = [0]*(N+1)
    for i, x in enumerate(a[:N+1]):
        if x:
            for j in range(N+1-i): c[i+j] += x*b[j]
    return c
def T(k):
    out = [0]*(K+1)
    for n in list(range(1, 40)) + list(range(-40, 0)):
        e = n*(3*n+1)//2 + k*n
        if n > 0:
            for m in range(0, K):
                ex = e + 5*n*m
                if ex > K: break
                if ex >= 0: out[ex] += (-1)**(n % 2)
        else:  # 1/(1-q^{5n}) = -sum_{m>=1} q^{5|n| m}
            for m in range(1, K):
                ex = e + 5*(-n)*m
                if ex > K: break
                if ex >= 0: out[ex] -= (-1)**(n % 2)
    return out
Ts = {k: T(k) for k in range(5)}
print("T2 == 0:", not any(Ts[2]), " T4 == -T0:", Ts[4] == [-x for x in Ts[0]], " T3 == -T1:", Ts[3] == [-x for x in Ts[1]])
# A, B from E dissection
def Epow(e, N=K, step=1):
    c = [0]*(N+1); c[0] = 1
    for n in range(1, N//step+1):
        for _ in range(abs(e)):
            if e > 0:
                for i in range(N, n*step-1, -1): c[i] -= c[i-n*step]
            else:
                for i in range(n*step, N+1): c[i] += c[i-n*step]
    return c
def inv(a, N):
    b = [0]*(N+1); b[0] = Fraction(1, a[0])
    for n in range(1, N+1): b[n] = -sum(a[k]*b[n-k] for k in range(1, n+1))/a[0]
    return b
E = Epow(1); M = K//5
e25 = Epow(1, K, 25)
ie = inv(e25, K)
A = mul([E[i] if i % 5 == 0 else 0 for i in range(K+1)], ie)       # A(q^5) as q-series
B = [-x for x in mul([E[i] if i % 5 == 2 else 0 for i in range(K+1)], ie)]
B = [0]*0 + B; B = [B[i+2] if i+2 <= K else 0 for i in range(K+1)]  # remove q^2
def P(*fs):
    r = [1]+[0]*K
    for f in fs: r = mul(r, f)
    return r
q = lambda e: [1 if i == e else 0 for i in range(K+1)]
def add(*ts):
    out = [0]*(K+1)
    for c, f in ts:
        for i in range(K+1): out[i] += c*f[i]
    return out
Qp = add((1, P(A,A,A,A)), (1, P(q(1),A,A,A)), (2, P(q(2),A,A)), (3, P(q(3),A)), (5, q(4)),
         (-3, P(q(5),B)), (2, P(q(6),B,B)), (-1, P(q(7),B,B,B)), (1, P(q(8),B,B,B,B)))
for k in (0, 1):
    c = mul(Ts[k], Qp)
    cls4 = [c[5*n+4] for n in range(M-2)]
    print(f"class4(T{k}·Q):", cls4[:12])
