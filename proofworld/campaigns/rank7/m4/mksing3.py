import pickle, sympy as sp
from mksing import gens
from core import EPs
allids=pickle.load(open('core_ids.pkl','rb'))
s='ring r=(0,Q),(J1,J2,J3,J4,J5,J6,J7,J8,J9,J10),dp;\noption(redSB);\n'
s+="ideal I="+",\n".join(gens)+";\n"
s+="ideal G=std(I);\npoly M=J7*J1;\n"
k=0; ids=[]
for c,den,nums in allids:
    for n in nums:
        n=sp.expand(sp.nsimplify(n))
        if n==0: continue
        degs={sp.degree(t,EPs) for t in sp.Add.make_args(n)}
        k+=1
        n=sp.expand(n.subs(EPs,1))
        ids.append((c,k,n,degs))
        s+=f"poly f{k}="+str(n).replace("**","^")+";\n"
        s+=f'print("class {c} id {k} EP{sorted(degs)}: ");print(reduce(f{k},G)==0);print(reduce(M*f{k},G)==0);\n'
open('t3.sing','w').write(s)
pickle.dump(ids,open('ids.pkl','wb'))
