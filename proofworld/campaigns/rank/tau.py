"""Global involution τ on ℤ² proving the reduced (2.19) identity for fixed a ≥ 0.
Points p=(U,V), exponent e = (3U²+V² − 4 − 24a²)/24.
 L-points: U = 2b−4a, V = 12r+6|b|+2 (r≥0, weight −1 on the L side ⇒ +… we use wD = wR − wL)
           U = 2b−4a, V = 12r+6|b|−2 (r≥1, wL = −1)
 R-points: U ≡ 1 (4), V ≡ 1 (6), V ≥ 6a+1 (wR = +1);  U ≡ 1 (4), V ≡ 5 (6), V ≥ 6a+5 (wR = −1)."""
def wL(a, U, V):
    if U % 2: return 0
    b = (U + 4*a)//2; B = abs(b)
    if V >= 6*B + 2 and (V - 6*B - 2) % 12 == 0: return 1
    if V >= 6*B + 10 and (V - 6*B + 2) % 12 == 0: return -1
    return 0
def wR(a, U, V):
    if U % 4 != 1: return 0
    if V % 6 == 1 and V >= 6*a + 1: return 1
    if V % 6 == 5 and V >= 6*a + 5: return -1
    return 0
def wD(a, U, V): return wR(a, U, V) - wL(a, U, V)
def rotp(U, V): return ((U + V)//2, (V - 3*U)//2)
def rotm(U, V): return ((U - V)//2, (V + 3*U)//2)
def nrm(U, V): return (U, V) if U % 4 == 1 else (-U, V)
def tau(a, U, V):
    if wL(a, U, V): return nrm(*rotp(U, V))
    if wR(a, U, V):
        for (u, v) in (rotm(U, V), rotm(-U, V)):
            if (U + V) % 2 == 0 and wL(a, u, v) and nrm(*rotp(u, v)) == (U, V): return (u, v)
        return nrm(*(rotp(U, V) if V % 4 == 1 else rotm(U, V)))
    return (U, V)
bad = 0
for a in range(0, 5):
    for U in range(-120, 121):
        for V in range(-60, 400):
            w = wD(a, U, V)
            if w == 0: continue
            t = tau(a, U, V)
            if tau(a, *t) != (U, V) or wD(a, *t) != -w or 3*t[0]**2 + t[1]**2 != 3*U*U + V*V:
                bad += 1
                if bad < 8: print("bad", a, (U, V), t, tau(a, *t), w, wD(a, *t))
print("violations:", bad)
