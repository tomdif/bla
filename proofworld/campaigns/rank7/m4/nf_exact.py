import pickle, sympy as sp, itertools
w=sp.Symbol('w')
PHI=sum(w**i for i in range(7))
def red(e):
    e=sp.expand(e)
    return sp.expand(sp.rem(sp.Poly(e,w),sp.Poly(PHI,w)).as_expr()) if e!=0 else 0
def c2(n): return n*(n-1)//2
def normth(e,k):
    k%=7; s=(e-e%3)//3; e0=e%3
    coef=(-1)**(s%2)*w**((-s*k)%7); qe=-(3*c2(s+1))-e0*s+3*s
    if e0==2: e0=1; k=(-k)%7
    if e0==0 and k==0: return None
    if e0==0 and k>3: coef=-coef*w**k; k=7-k
    return coef,qe,(e0,k)
ATOMS=[(0,1),(0,2),(0,3)]+[(1,k) for k in range(7)]
AI={a:i for i,a in enumerate(ATOMS)}
def prod(args):
    co=1; qe=0; J=[0]*len(ATOMS)
    for (e,k) in args:
        t=normth(e,k)
        if t is None: return None
        co*=t[0]; qe+=t[1]; J[AI[t[2]]]+=1
    return co,qe,tuple(J)
def inst(x,y,u,v):
    add=lambda a,b:(a[0]+b[0],a[1]+b[1]); sub=lambda a,b:(a[0]-b[0],a[1]-b[1])
    A=prod([add(x,y),sub(x,y),add(u,v),sub(u,v)]); B=prod([add(x,v),sub(x,v),add(u,y),sub(u,y)])
    C=prod([add(y,v),sub(y,v),add(x,u),sub(x,u)])
    uy=sub(u,y); poly={}
    for t,s,qs,zs in ((A,1,0,0),(B,-1,0,0),(C,-1,uy[0],uy[1])):
        if t is None: continue
        key=(t[2],t[1]+qs); poly[key]=poly.get(key,0)+s*t[0]*w**(zs%7)
    return {k:red(v) for k,v in poly.items() if red(v)!=0}
insts=[((-2,0),(-2,1),(0,0),(0,2)),((-2,0),(-2,2),(0,1),(0,2)),((-2,1),(-2,6),(0,1),(0,2)),((-2,0),(-2,6),(0,0),(0,2)),((-2,1),(-2,2),(0,0),(0,2))]
mults=[((-1,-1,0,0,1,0,1,1,0,0),-1),((-1,-1,0,0,2,0,0,1,0,0),-1),((-1,-1,0,-1,0,1,1,0,0,2),-1),((-1,-1,0,0,0,0,2,0,0,1),-1),((-1,-1,0,0,1,0,0,1,0,1),-1)]
# NOTE: mults were relative to normalized instances; recompute raw instance rows and let solver find lambdas with mults adjusted
num,den,TH=pickle.load(open('nf_rem.pkl','rb'))
qq,kap=sp.symbols('q kappa')
tgt={}
for t in sp.Add.make_args(sp.expand(num/kap)):
    c,m=t.as_coeff_Mul(); d=m.as_powers_dict()
    J=[0]*len(ATOMS)
    for (e,k),s in TH.items(): J[AI[(e,k)]]=int(d.get(s,0))
    key=(tuple(J),int(d.get(qq,0)))
    tgt[key]=red(tgt.get(key,0)+c*w**int(d.get(w,0)))
# find for each raw instance the monomial multiplier aligning to target: try all multipliers sig/tau
rows=[]
for x,y,u,v in insts:
    W=inst(x,y,u,v)
    for tau in W:
        for sig in tgt:
            m=(tuple(a-b for a,b in zip(sig[0],tau[0])),sig[1]-tau[1])
            rows.append(((x,y,u,v),m,{(tuple(a+b for a,b in zip(k[0],m[0])),k[1]+m[1]):c for k,c in W.items()}))
keys=sorted({k for r in rows for k in r[2]}|set(tgt))
lam=sp.symbols('l0:%d'%len(rows))
eqs=[]
for k in keys:
    e=sum(lam[i]*r[2].get(k,0) for i,r in enumerate(rows))-tgt.get(k,0)
    eqs.append(sp.expand(e))
# solve over Q(w): treat coefficients as polys in w; unknowns linear; use coefficient matching after writing lam_i = sum a_ij w^j
A={}
unk=[]
for i in range(len(rows)):
    for j in range(6): unk.append(sp.Symbol(f'a{i}_{j}'))
sub={lam[i]:sum(sp.Symbol(f'a{i}_{j}')*w**j for j in range(6)) for i in range(len(rows))}
lin=[]
for e in eqs:
    e=red(sp.expand(e.subs(sub)))
    if e==0: continue
    P=sp.Poly(e,w)
    for cf in P.all_coeffs(): lin.append(cf)
sol=sp.solve(lin,unk,dict=True)
print("solutions:",len(sol))
s0=sol[0]
used=[]
for i,r in enumerate(rows):
    val=red(sum(s0.get(sp.Symbol(f'a{i}_{j}'),sp.Symbol(f'a{i}_{j}'))*w**j for j in range(6)))
    val=sp.sympify(val).subs({u_:0 for u_ in unk})
    if val!=0: used.append((r[0],r[1],val)); print(r[0],r[1],val)
pickle.dump((used,tgt),open('nf_exact.pkl','wb'))
