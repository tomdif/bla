import pickle, sys, time
import sparsecert as SC
from sparsecert import div, mul, P_
raw,res=pickle.load(open(sys.argv[1] if len(sys.argv)>1 else 'coords2.pkl','rb'))
INST0=list(SC.INST)
proved={k:v for k,v in res.items() if v[0]}
todo=[k for k,v in res.items() if not v[0]]
print("start proved",len(proved),"todo",len(todo))
changed=True; rnd=0
while changed and todo:
    changed=False; rnd+=1
    # generator list: instances + proved targets (as dict polys)
    extra=[dict((k,int(v.p*pow(int(v.q),P_-2,P_))%P_) for k,v in proved[key][2].items()) for key in sorted(proved)]
    SC.INST=INST0+extra
    for key in list(todo):
        tgt=res[key][2]
        ok,n,cert=SC.solve(tgt,rounds=1)
        print(rnd,key,ok,len(cert),flush=True)
        if ok:
            # tag which generators are proved-identities
            res[key]=(ok,cert,tgt); proved[key]=res[key]; todo.remove(key); changed=True
            res[key]=(ok,{('P' if i>=len(INST0) else 'W', i if i<len(INST0) else sorted(proved)[i-len(INST0)] if False else i, m):v for (i,m),v in cert.items()},tgt,len(INST0),sorted(k for k in proved if k!=key))
print("remaining",todo)
pickle.dump((raw,res),open('fix_'+(sys.argv[1] if len(sys.argv)>1 else 'coords2.pkl'),'wb'))
