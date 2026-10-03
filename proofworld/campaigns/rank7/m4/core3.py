import pickle, sympy as sp
from core import Qs,EPs,Jsym,Ksym,ev
res=pickle.load(open('core_res.pkl','rb'))
allids=[]
for c in range(7):
    exprs=[x for x in res[c] if x!=0]
    # common denominator per class
    den=sp.lcm([sp.fraction(sp.together(x))[1] for x in exprs])
    nums=[sp.expand(sp.cancel(x*den)) for x in exprs]
    mons=sorted({m for n in nums for m in sp.Add.make_args(n) for m in [m.as_coeff_Mul()[1]]},key=str)
    M=sp.Matrix([[sp.expand(n).coeff(m) if False else 0 for m in mons] for n in nums])
    rows=[]
    for n in nums:
        d=dict()
        for term in sp.Add.make_args(n):
            cf,mm=term.as_coeff_Mul(); d[mm]=d.get(mm,0)+cf
        rows.append([d.get(m,0) for m in mons])
    M=sp.Matrix(rows)
    print("class",c,"ids",len(nums),"rank",M.rank(),"monomials",len(mons),"den",den)
    allids.append((c,den,nums))
pickle.dump(allids,open('core_ids.pkl','wb'))
c,den,nums=allids[1]
for n in nums: print(sp.factor(n))
