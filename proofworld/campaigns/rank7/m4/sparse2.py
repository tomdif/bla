import pickle, sys, time
from sparsecert import *
def solve2(target):
    S0=set(target); allc=[]; seen=set()
    c1=candidates(S0)
    for c in c1: seen.add(c)
    allc+=c1
    S1=set(S0)
    for c in c1: S1|=set(cand_row(*c))
    new=S1-S0
    c2=[]
    for i,W in enumerate(INST):
        for tau in W:
            for sig in new:
                m=div(sig,tau); c=(i,m)
                if c in seen: continue
                row=cand_row(i,m)
                if sum(1 for k in row if k in S1)>=2:
                    seen.add(c); c2.append(c)
    allc+=c2
    return allc
def elim(allc,target):
    piv={}
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
    return (not rest), {allc[k]:(-v)%P_ for k,v in comb.items()}
if __name__=="__main__":
    raw=pickle.load(open('raw.pkl','rb'))
    res=pickle.load(open('certs.pkl','rb'))
    out={}
    for c in range(7):
        for i in range(6):
            if raw[c][i]==0 or res[(c,i)][0]: continue
            t0=time.time()
            tgt=sym2dict(raw[c][i])
            allc=solve2(tgt)
            ok,cert=elim(allc,tgt)
            out[(c,i)]=(ok,cert,tgt)
            print(c,i,"cands",len(allc),"found",ok,"cert",len(cert),"%.0fs"%(time.time()-t0),flush=True)
            pickle.dump(out,open('certs2.pkl','wb'))
