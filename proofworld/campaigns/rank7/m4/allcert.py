import pickle, time
from sparsecert import *
raw=pickle.load(open('raw.pkl','rb'))
res={}
for c in range(7):
    for i in range(6):
        if raw[c][i]==0: continue
        t0=time.time()
        tgt=sym2dict(raw[c][i])
        ok,n,cert=solve(tgt,rounds=1)
        res[(c,i)]=(ok,cert,tgt)
        print(c,i,"terms",len(tgt),"found",ok,"cands",n,"cert",len(cert),"%.1fs"%(time.time()-t0),flush=True)
pickle.dump(res,open('certs.pkl','wb'))
