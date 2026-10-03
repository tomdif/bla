import pickle, itertools
from fractions import Fraction as Fr
from winst import inst, normalize
from sparsecert import INST, P_
L=pickle.load(open('winst.pkl','rb'))
# parameter search
PARAM={}
for x,y in sorted(itertools.product(range(-21,22),repeat=2),key=lambda p:(max(abs(p[0]),abs(p[1])),abs(p[0])+abs(p[1]))):
    for u in range(1,11):
        for v in range(u+1,11):
            n=normalize(inst(2*x,2*y,2*u,2*v))
            if n in set(L) and n not in PARAM: PARAM[n]=(x,y,u,v)
def ratrec(a):
    a%=P_
    for den in range(1,50):
        num=a*den%P_
        if num<1000: return Fr(num,den)
        if P_-num<1000: return Fr(num-P_,den)
    raise ValueError(a)
def c2(k): return k*(k-1)//2
def normexp(e):
    r=e%21; j=(e-r)//21
    sgn=(-1)**(j%2); m=-21*c2(j+1)-r*j+21*j
    if r==0: return None
    R=r if r<=10 else 21-r
    return r,j,sgn,m,R
def lit(z): return f"({z})" if z<0 else f"{z}"
def mon(J,qe,first=True):
    parts=[]
    if qe>0: parts.append(f"Qt hd ^ {qe}")
    if qe<0: parts.append(f"(Qt hd ^ {-qe})⁻¹")
    for a,e in enumerate(J):
        if e>0: parts.append(f"Jt hN hd {a+1}" + (f" ^ {e}" if e>1 else ""))
        if e<0: parts.append(f"(Jt hN hd {a+1} ^ {-e})⁻¹")
    return " * ".join(parts) if parts else "1"
def term(c,J,qe):
    c=Fr(c)
    m=mon(J,qe)
    cs=f"({c.numerator}/{c.denominator} : L)" if c.denominator!=1 else f"({c.numerator} : L)"
    return f"{cs} * {m}"
def gen(name,tgt,cert):
    out=[]; elemmas={}
    wl=[]
    for k,((i,m),lam) in enumerate(cert.items()):
        n=L[i]; x,y,u,v=PARAM[n]
        exprs=[(x,y,'+'),(x,y,'-'),(u,v,'+'),(u,v,'-'),(x,v,'+'),(x,v,'-'),(u,y,'+'),(u,y,'-'),(y,v,'+'),(y,v,'-'),(x,u,'+'),(x,u,'-')]
        for a,b,op in exprs:
            e=a+b if op=='+' else a-b
            key=f"{lit(a)} {op} {lit(b)}"
            elemmas[key]=e
        # raw = s*Q^{mq}*Wnorm : recompute
        raw=inst(2*x,2*y,2*u,2*v)
        mq=min(kk[1] for kk in raw); p={(kk[0],kk[1]-mq):vv for kk,vv in raw.items()}
        ks=sorted(p); s=1 if p[ks[0]]>0 else -1
        assert tuple(sorted((kk,s*vv) for kk,vv in p.items()))==n
        # coefficient for w: lam * m * s * Q^{-mq}
        J,qe=m
        wl.append((k,x,y,u,v,ratrec(lam)*s,J,qe-mq))
    out.append(f"theorem {name} :")
    out.append("    "+" + ".join(term(c,J,qe) for (J,qe),c in tgt.items())+" = 0 := by")
    out.append("  have hQ := Qt_ne hd")
    for a in range(1,11):
        out.append(f"  have hJ{a} := Jt_ne hN hd (a := {a}) (by norm_num) (by norm_num)")
    names=[]
    for idx,(key,e) in enumerate(elemmas.items()):
        ne=normexp(e)
        nm=f"e{idx}"; names.append(nm)
        lhs=f"θ hN (↑d * ({key})) 1"
        if ne is None:
            j=e//21
            out.append(f"  have {nm} : {lhs} = 0 := Jt_norm0 hN hd _ {lit(j)} (by norm_num)")
        else:
            r,j,sgn,m,R=ne
            sg="1" if sgn>0 else "-1"
            qp=(f"Qt hd ^ {m} * " if m>0 else (f"(Qt hd ^ {-m})⁻¹ * " if m<0 else ""))
            rhs=("-" if sgn<0 else "")+"("+qp+f"Jt hN hd {R})"
            h3="Or.inl rfl" if r==R else "Or.inr (by norm_num)"
            out.append(f"  have {nm} : {lhs} = {rhs} := by")
            out.append(f"    rw [Jt_norm hN hd _ {r} {lit(j)} ({m}) {R} ({sg}) (by norm_num) (by norm_num [c2]) ({h3}) (by norm_num)]")
            out.append(f"    simp only [zpow_neg, zpow_ofNat, zpow_one, zpow_zero, one_mul]; ring")
    for (k,x,y,u,v,c,J,qe) in wl:
        out.append(f"  have w{k} := weierQ hN hd (x := {x}) (y := {y}) (u := {u}) (v := {v}) (by norm_num) (by norm_num) (by norm_num)")
        out.append(f"  simp only [{', '.join(names)}] at w{k}")
        kk=u-y
        if kk>=0: out.append(f"  rw [show (({u} : ℤ) - {lit(y)}) = (({kk} : ℕ) : ℤ) by norm_num, zpow_natCast] at w{k}")
        else: out.append(f"  rw [show (({u} : ℤ) - {lit(y)}) = -(({-kk} : ℕ) : ℤ) by norm_num, zpow_neg, zpow_natCast] at w{k}")
    comb=" + ".join(f"{term(c,J,qe)} * w{k}" for (k,x,y,u,v,c,J,qe) in wl)
    out.append(f"  linear_combination (norm := skip) {comb}")
    out.append("  field_simp")
    out.append("  ring")
    return "\n".join(out)
if __name__=="__main__":
    res=pickle.load(open('certs.pkl','rb'))
    ok,cert,tgt=res[(1,1)]
    print(gen("cubic_test",tgt,cert))
