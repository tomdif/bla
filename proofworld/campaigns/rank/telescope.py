import sympy as sp
q = sp.symbols('q')
def poch(lo, m): return sp.prod([1 - q**(lo+i) for i in range(m)]) if m > 0 else sp.Integer(1)
def alpha(r, k):
    return q**(2*r*r+2*k*r+r) - (q**(2*r*r+2*k*r-r-k) if r >= 1 else 0)
def term(n, r, k): return alpha(r, k) / (poch(1, n-r) * poch(k+1, n+r))
for k in (0, 1, 2):
    for n in (2, 3, 4):
        # difference summand s(r) = term(n,r) - term(n-1,r)   (term(n-1,n) := 0)
        s = [sp.simplify(term(n, r, k) - (term(n-1, r, k) if r <= n-1 else 0)) for r in range(n+1)]
        target = q**n / (poch(1, n) * poch(k+1, n))
        assert sp.simplify(sum(s) - target) == 0
        # tails T(r) = sum_{i>=r} s(i); normalise by A_r/((q)_{n-r}(q^{k+1})_{n+r-1})
        out = []
        for r in range(1, n+1):
            Tr = sp.factor(sp.simplify(sum(s[r:])))
            norm = sp.factor(sp.simplify(Tr * poch(1, n-r) * poch(k+1, n+r-1) / q**(2*r*r+2*k*r-r-k)))
            out.append(norm)
        print(f"k={k} n={n}:", out)
