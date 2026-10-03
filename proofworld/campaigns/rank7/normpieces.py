exec(open('consist7.py').read().split("E2 = {}")[0])
for key, e in sorted(G.items()):
    e = sp.expand(e)
    terms = {}
    for ph in (P1, P2, P3):
        cf = sp.factor(sp.simplify(e.coeff(ph)))
        if cf != 0: terms[str(ph)] = cf
    th = sp.factor(sp.simplify(e.subs({P1: 0, P2: 0, P3: 0})))
    print(key, terms, "| theta:", th)
