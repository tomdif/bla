import pickle
from fractions import Fraction as Fr
from genlean import *
from winst import inst
res=pickle.load(open('certs.pkl','rb'))
def check(tgt,cert):
    acc={k:Fr(v.p,v.q) if hasattr(v,'p') else Fr(v) for k,v in tgt.items()}
    for k,((i,m),lam) in enumerate(cert.items()):
        n=L[i]; x,y,u,v=PARAM[n]
        raw=inst(2*x,2*y,2*u,2*v)
        mq=min(kk[1] for kk in raw); p={(kk[0],kk[1]-mq):vv for kk,vv in raw.items()}
        ks=sorted(p); s=1 if p[ks[0]]>0 else -1
        c=ratrec(lam)*s; J,qe=m; qe-=mq
        for (JJ,qq),vv in raw.items():
            key=(tuple(a+b for a,b in zip(JJ,J)),qq+qe)
            acc[key]=acc.get(key,0)-c*vv
    return {k:v for k,v in acc.items() if v}
ok,cert,tgt=res[(1,1)]
print(check(tgt,cert))
