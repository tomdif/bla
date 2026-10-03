import numpy as np
P_=2147483629  # prime < 2^31
def rref(M):
    M=M.copy()%P_; r=0; piv=[]
    rows,cols=M.shape
    for c in range(cols):
        nz=np.nonzero(M[r:,c])[0]
        if len(nz)==0: continue
        i=r+nz[0]
        if i!=r: M[[r,i]]=M[[i,r]]
        inv=pow(int(M[r,c]),P_-2,P_)
        M[r]=(M[r]*inv)%P_
        col=M[:,c].copy(); col[r]=0
        nzr=np.nonzero(col)[0]
        if len(nzr):
            M[nzr]=(M[nzr]-(col[nzr,None]*M[r][None,:])%P_)%P_
        piv.append(c); r+=1
        if r==rows: break
    return M[:r],piv
def reduce(v,B,piv):
    v=v.copy()%P_
    for i,c in enumerate(piv):
        if v[c]: v=(v-(v[c]*B[i])%P_)%P_
    return v
