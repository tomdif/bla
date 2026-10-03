import pickle, sympy as sp, sys
from core import Qs,Jsym,EPs
P_=2147483629
inst=pickle.load(open('winst.pkl','rb'))   # list of tuples ((J,qe),c)
# monomial key: (J tuple len10 (can be negative), qe)
def mk(J,qe): return (tuple(J),qe)
def div(a,b): return (tuple(x-y for x,y in zip(a[0],b[0])),a[1]-b[1])
def mul(a,b): return (tuple(x+y for x,y in zip(a[0],b[0])),a[1]+b[1])
def sym2dict(expr):
    expr=sp.expand(sp.nsimplify(expr).subs(EPs,1)); out={}
    for term in sp.Add.make_args(expr):
        c,m=term.as_coeff_Mul(); d=m.as_powers_dict()
        key=(tuple(int(d.get(Jsym[a],0)) for a in range(1,11)),int(d.get(Qs,0)))
        out[key]=out.get(key,0)+sp.Rational(c)
    return out
INST=[dict(p) for p in inst]
def candidates(S):
    cands={}
    for i,W in enumerate(INST):
        for tau in W:
            for sig in S:
                m=div(sig,tau)
                cands[(i,m)]=1
    return list(cands)
def cand_row(i,m):
    return {mul(k,m):v for k,v in INST[i].items()}
def solve(target,rounds=2,maxc=200000):
    S=set(target); allc=[]; seen=set()
    for r in range(rounds):
        new=[c for c in candidates(S) if c not in seen]
        for c in new: seen.add(c)
        allc+=new
        for c in new: S|=set(cand_row(*c))
        if len(allc)>maxc: break
    # sparse elimination mod p with combination tracking
    piv={}  # col -> (row dict, comb dict)
    def reduce(row,comb):
        row=dict(row); comb=dict(comb)
        while True:
            cols=[c for c in row if c in piv]
            if not cols: return row,comb
            c=cols[0]; f=row[c]; prow,pcomb=piv[c]
            for k,v in prow.items():
                nv=(row.get(k,0)-f*v)%P_
                if nv: row[k]=nv
                else: row.pop(k,None)
            for k,v in pcomb.items():
                nv=(comb.get(k,0)-f*v)%P_
                if nv: comb[k]=nv
                else: comb.pop(k,None)
    for idx,c in enumerate(allc):
        row={k:v%P_ for k,v in cand_row(*c).items()}
        row,comb=reduce(row,{idx:1})
        if row:
            pc=min(row); inv=pow(row[pc],P_-2,P_)
            row={k:v*inv%P_ for k,v in row.items()}; comb={k:v*inv%P_ for k,v in comb.items()}
            piv[pc]=(row,comb)
    t={k:int(v.p*pow(int(v.q),P_-2,P_))%P_ for k,v in target.items()}
    rest,comb=reduce(t,{})
    return (not rest), len(allc), {allc[k]:(-v)%P_ for k,v in comb.items()}
if __name__=="__main__":
    raw=pickle.load(open('raw.pkl','rb'))
    c,i=int(sys.argv[1]),int(sys.argv[2])
    tgt=sym2dict(raw[c][i])
    ok,n,cert=solve(tgt,rounds=int(sys.argv[3]) if len(sys.argv)>3 else 1)
    print(c,i,"terms",len(tgt),"found",ok,"cands",n,"cert size",len(cert))
