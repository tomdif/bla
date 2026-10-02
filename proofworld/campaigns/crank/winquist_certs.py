import sympy as sp
N,M,c1,c2,u,v,s,t = sp.symbols('N M c1 c2 u v s t')
T2 = lambda m: m*(m-1)          # 2*T
P2 = lambda m: m*(3*m-1)        # 2*P
eF2 = lambda s,t: T2(N-s-t)+T2(M-s+t)+T2(s)+T2(t)
def cert(expr, rels):
    # expr must lie in ideal(rels) (rels linear); return cofactors
    G = sp.groebner(rels, c1, c2, N, M, u, v, s, t, order='lex')
    q, r = sp.reduced(sp.expand(expr), rels, c1, c2, N, M, u, v, s, t, order='lex')
    assert sp.expand(r) == 0, r
    return q
for r1 in (0,1):
  for r2 in (0,2):
    h1 = 3*c1 - (N+M-r1)
    h2 = 3*c2 - (N-M+(1 if r2==2 else 0))
    S = c1 + (u if r1==1 else -u); Tt = c2 + (v if r2==0 else -v)
    q = cert(eF2(S,Tt) - (P2(u)+P2(v)+eF2(c1,c2)), [h1,h2])
    w = cert(6*eF2(c1,c2) - 2*(N**2+M**2-3*N-M), [h1,h2])
    print(f"good r1={r1} r2={r2}: exp cofactors {[sp.factor(x) for x in q]}  eW cofactors {[sp.expand(x) for x in w]}")
h1 = 3*c1-(N+M-2); print("bad1:", cert(eF2(2*c1+1-s,t)-eF2(s,t),[h1]))
h2 = 3*c2-(N-M-1); print("bad2:", cert(eF2(s,2*c2+1-t)-eF2(s,t),[h2]))
