import pickle, time, sys
P_=2147483629
GENS=pickle.load(open('mixed_gens.pkl','rb'))
GL=list(GENS.keys()); GP=[dict(g) for g in GL]
targets,_=pickle.load(open('mixed_targets.pkl','rb'))
def div(a,b): return (tuple(x-y for x,y in zip(a[0],b[0])),a[1]-b[1])
def mul(a,b): return (tuple(x+y for x,y in zip(a[0],b[0])),a[1]+b[1])
def cands(S,seen):
    out=[]
    for i,W in enumerate(GP):
        for tau in W:
            for sig in S:
                c=(i,div(sig,tau))
                if c not in seen: seen.add(c); out.append(c)
    return out
def row(c):
    i,m=c; return {mul(k,m):v%P_ for k,v in GP[i].items()}
def solve(target,rounds=1):
    S=set(target); seen=set(); allc=[]
    for r in range(rounds):
        new=cands(S,seen); allc+=new
        for c in new: S|=set(row(c))
    piv={}
    def reduce(rw,comb):
        rw=dict(rw); comb=dict(comb)
        while True:
            cols=[c for c in rw if c in piv]
            if not cols: return rw,comb
            c=cols[0]; f=rw[c]; prow,pcomb=piv[c]
            for k,v in prow.items():
                nv=(rw.get(k,0)-f*v)%P_
                if nv: rw[k]=nv
                else: rw.pop(k,None)
            for k,v in pcomb.items():
                nv=(comb.get(k,0)-f*v)%P_
                if nv: comb[k]=nv
                else: comb.pop(k,None)
    for idx,c in enumerate(allc):
        rw,comb=reduce(row(c),{idx:1})
        if rw:
            pc=min(rw); inv=pow(rw[pc],P_-2,P_)
            piv[pc]=({k:v*inv%P_ for k,v in rw.items()},{k:v*inv%P_ for k,v in comb.items()})
    t={k:int(v.p*pow(int(v.q),P_-2,P_))%P_ for k,v in target.items()}
    rest,comb=reduce(t,{})
    return (not rest), len(allc), {allc[k]:(-v)%P_ for k,v in comb.items()}
if __name__=="__main__":
    keys=[tuple(map(int,a.split(','))) for a in sys.argv[1:]] or sorted(targets)
    res={}
    for key in keys:
        t0=time.time(); ok,n,cert=solve(targets[key])
        res[key]=(ok,cert)
        print(key,"terms",len(targets[key]),"found",ok,"cands",n,"cert",len(cert),"%.1fs"%(time.time()-t0),flush=True)
    pickle.dump(res,open('mixed_certs.pkl','wb'))
