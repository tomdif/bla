import pickle, sys
from fractions import Fraction as Fr
from genlean import PARAM, ratrec, normexp, lit, mon, term, L
from winst import inst
raw_,res=pickle.load(open(sys.argv[1],'rb'))
prefix=sys.argv[2]
used={}
for key,(ok,cert,tgt) in sorted(res.items()):
    assert ok
    for (i,m),lam in cert.items():
        n=L[i]; used[PARAM[n]]=n
def iname(p): return "W_"+"_".join(("m%d"%(-v) if v<0 else "%d"%v) for v in p)
out=[]
# instance lemmas
for p in sorted(used):
    x,y,u,v=p
    raw=inst(2*x,2*y,2*u,2*v)
    out.append(f"theorem {iname(p)} :")
    out.append("    "+" + ".join(term(c,J,qe) for (J,qe),c in raw.items())+" = 0 := by")
    out.append("  have hQ := Qt_ne hd")
    elem={}
    for a,b,op in [(x,y,'+'),(x,y,'-'),(u,v,'+'),(u,v,'-'),(x,v,'+'),(x,v,'-'),(u,y,'+'),(u,y,'-'),(y,v,'+'),(y,v,'-'),(x,u,'+'),(x,u,'-')]:
        elem[f"{lit(a)} {op} {lit(b)}"]=a+b if op=='+' else a-b
    names=[]
    for idx,(key,e) in enumerate(elem.items()):
        ne=normexp(e); nm=f"e{idx}"; names.append(nm)
        lhs=f"θ hN (↑d * ({key})) 1"
        if ne is None:
            out.append(f"  have {nm} : {lhs} = 0 := Jt_norm0 hN hd _ {lit(e//21)} (by norm_num)")
        else:
            r,j,sgn,mm,R=ne; sg="1" if sgn>0 else "-1"
            qp=(f"Qt hd ^ {mm} * " if mm>0 else (f"(Qt hd ^ {-mm})⁻¹ * " if mm<0 else ""))
            rhs=("-" if sgn<0 else "")+"("+qp+f"Jt hN hd {R})"
            h3="Or.inl rfl" if r==R else "Or.inr (by norm_num)"
            out.append(f"  have {nm} : {lhs} = {rhs} := by")
            out.append(f"    rw [Jt_norm hN hd _ {r} {lit(j)} ({mm}) {R} ({sg}) (by norm_num) (by norm_num [c2]) ({h3}) (by norm_num)]")
            out.append(f"    simp only [zpow_neg, zpow_ofNat, zpow_one, zpow_zero, one_mul]; ring")
    out.append(f"  have w := weierQ hN hd (x := {x}) (y := {y}) (u := {u}) (v := {v}) (by norm_num) (by norm_num) (by norm_num)")
    out.append(f"  simp only [{', '.join(names)}] at w")
    kk=u-y
    if kk>=0: out.append(f"  rw [show (({u} : ℤ) - {lit(y)}) = (({kk} : ℕ) : ℤ) by norm_num, zpow_natCast] at w")
    else: out.append(f"  rw [show (({u} : ℤ) - {lit(y)}) = -(({-kk} : ℕ) : ℤ) by norm_num, zpow_neg, zpow_natCast] at w")
    for a in range(1,11):
        out.append(f"  have hJ{a} := Jt_ne hN hd (a := {a}) (by norm_num) (by norm_num)")
    out.append(f"  linear_combination (norm := skip) w")
    out.append(f"  field_simp")
    out.append(f"  ring")
    out.append("")
# core lemmas
for key,(ok,cert,tgt) in sorted(res.items()):
    c,i=key
    out.append(f"theorem {prefix}_{c}_{i} :")
    out.append("    "+" + ".join(term(v,J,qe) for (J,qe),v in tgt.items())+" = 0 := by")
    out.append("  have hQ := Qt_ne hd")
    for a in range(1,11):
        out.append(f"  have hJ{a} := Jt_ne hN hd (a := {a}) (by norm_num) (by norm_num)")
    comb=[]
    for (i_,m),lam in cert.items():
        n=L[i_]; p=PARAM[n]
        rawp=inst(2*p[0],2*p[1],2*p[2],2*p[3])
        mq=min(kk[1] for kk in rawp); pp={(kk[0],kk[1]-mq):vv for kk,vv in rawp.items()}
        ks=sorted(pp); s=1 if pp[ks[0]]>0 else -1
        cc=ratrec(lam)*s; J,qe=m
        comb.append(f"{term(cc,J,qe-mq)} * {iname(p)} hN hd")
    out.append(f"  linear_combination (norm := skip) "+" + ".join(comb))
    out.append("  field_simp")
    out.append("  ring")
    out.append("")
print("\n".join(out))
print("-- instances:",len(used),file=sys.stderr)
