import pickle, sympy as sp
from mksing import gens
allids=pickle.load(open('core_ids.pkl','rb'))
s='LIB "elim.lib";\nring r=(0,Q),(J1,J2,J3,J4,J5,J6,J7,J8,J9,J10),dp;\noption(redSB);\n'
s+="ideal I="+",\n".join(gens)+";\n"
s+="ideal G=std(I);\n"
s+="list L=sat(G,J1*J2*J3*J4*J5*J6*J7*J8*J9*J10);\nideal S=L[1];\nprint(size(S));print(L[2]);\n"
k=0
for c,den,nums in allids:
    for n in nums:
        n=sp.nsimplify(n)
        if n==0: continue
        k+=1
        s+=f"poly f{k}="+str(sp.expand(n)).replace("**","^")+";\n"
        s+=f'print("class {c} id {k}: G="+string(reduce(f{k},G)==0)+" S="+string(reduce(f{k},S)==0));\n'
open('t2.sing','w').write(s)
