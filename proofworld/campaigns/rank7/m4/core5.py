import pickle, sympy as sp, itertools
allids=pickle.load(open('core_ids.pkl','rb'))
def core(f):
    f=sp.factor(f)
    fl=sp.factor_list(f)[1]
    return sp.expand(sp.Mul(*[fa**e for fa,e in fl if len(sp.Add.make_args(fa))>1])), sp.Mul(*[fa**e for fa,e in fl if len(sp.Add.make_args(fa))==1])
for c,den,nums in allids:
    nums=[sp.nsimplify(n) for n in nums if n!=0]
    # try all pairwise combos a*n1+b*n2 with small ints to find small cores
    best={}
    for i in range(len(nums)):
        for j in range(len(nums)):
            for a,b in itertools.product(range(-3,4),repeat=2):
                if (a,b)==(0,0): continue
                f=sp.expand(a*nums[i]+b*nums[j])
                if f==0: continue
                cr,mon=core(f)
                n=len(sp.Add.make_args(cr))
                key=str(cr)
                if n<=6: best[key]=(n,cr)
    print("class",c, sorted([(v[0],k) for k,v in best.items()])[:6])
