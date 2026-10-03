import pickle, sympy as sp
from core import Qs,EPs,Jsym,Ksym,ev
allids=pickle.load(open('core_ids.pkl','rb'))
seen=[]
for c,den,nums in allids:
    for n in nums:
        f=sp.factor(sp.nsimplify(n))
        # strip monomial factors
        core=sp.Mul(*[fa**e for fa,e in sp.factor_list(f)[1] if len(sp.Add.make_args(fa))>1])
        core=sp.expand(core)
        key=str(core)
        if key not in seen:
            seen.append(key); print(c, len(sp.Add.make_args(core)), core)
