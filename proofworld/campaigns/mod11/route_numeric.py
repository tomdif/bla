"""Exact-arithmetic check of every identity in the triTheta^3 route before formalizing (planted truth)."""
K = 120                                   # q-degree checked
def Tp(n): return n * (n - 1) // 2        # n(n-1)/2, the triTheta exponent
def E_pow(e, N=K):
    c = [0] * (N + 1); c[0] = 1
    for n in range(1, N + 1):
        for _ in range(e):
            for i in range(N, n - 1, -1): c[i] -= c[i - n]
    return c
def mul(a, b):
    c = [0] * (K + 1)
    for i, x in enumerate(a):
        if x:
            for j in range(K + 1 - i): c[i + j] += x * b[j]
    return c
B = 20
R = {}                                    # R_N(q) = sum_{n1+n2+n3=N} q^{sum Tp(n_i)}
for n1 in range(-B, B + 1):
    for n2 in range(-B, B + 1):
        for n3 in range(-B, B + 1):
            e = Tp(n1) + Tp(n2) + Tp(n3)
            if e <= K:
                R.setdefault(n1 + n2 + n3, [0] * (K + 1))[e] += 1
Rget = lambda N: R.get(N, [0] * (K + 1))
shift = lambda s, k: [0] * k + s[:K + 1 - k] if k >= 0 else None
E, E3, E9, E10 = E_pow(1), E_pow(3), E_pow(9), E_pow(10)
# (1) 6 E^9 = sum_N w3(N) R_N,  w3(N) = d^3/dz^3 z^N at z=-1 = N(N-1)(N-2)(-1)^(N-3)
w3 = lambda N: N * (N - 1) * (N - 2) * (-1) ** ((N - 3) % 2)
lhs = [0] * (K + 1)
for N, s in R.items():
    for k in range(K + 1): lhs[k] += w3(N) * s[k]
print("(1) 6E^9 = Σ w3(N) R_N         :", lhs[:60] == [6 * x for x in E9][:60])   # box B limits accuracy
# (2) quasi-periodicity R_{N+3} = q^N R_N ; reflection R_{3-N} = R_N
ok2 = all(Rget(N + 3)[k] == (Rget(N)[k - N] if 0 <= k - N else 0) for N in range(-6, 7) for k in range(40))
ok2b = all(Rget(3 - N)[k] == Rget(N)[k] for N in range(-6, 7) for k in range(40))
print("(2) R_{N+3} = q^N R_N           :", ok2, "| R_{3-N} = R_N:", ok2b, "| so R_2 = R_1:", Rget(2)[:40] == Rget(1)[:40])
# (3) E*R_0 and E*R_1 as explicit series (3-dissection of Jacobi's E^3 = Σ(-1)^n (2n+1) q^{n(n+1)/2})
ER0, ER1 = mul(E, Rget(0)), mul(E, Rget(1))
Bser = [0] * (K + 1)                      # Σ_j (-1)^j (6j+1) q^{j(3j+1)/2}
for j in range(-30, 31):
    e = j * (3 * j + 1) // 2
    if 0 <= e <= K: Bser[e] += (-1) ** (j % 2) * (6 * j + 1)
C3 = [0] * (K + 1)                        # Σ_{i≥0} (-1)^i (2i+1) q^{3i(i+1)/2} = E(q^3)^3
for i in range(0, 30):
    e = 3 * i * (i + 1) // 2
    if e <= K: C3[e] += (-1) ** (i % 2) * (2 * i + 1)
print("(3) E·R_0 = Bser                :", ER0[:60] == Bser[:60], "  E·R_0 =", ER0[:8])
print("    E·R_1 = 3·C3 ?             :", ER1[:60] == [3 * x for x in C3][:60], "  E·R_1 =", ER1[:8], " C3 =", C3[:8])
# (4) the mod-11 vanishing of E^10 at exponents ≡ 6
print("(4) [q^a]E^10 ≡ 0 (11), a≡6     :", all(E10[a] % 11 == 0 for a in range(6, K + 1, 11)))
