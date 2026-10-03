# mixed-level certificate search: atoms J1..J10 (level 21), K1..K3 (theta(Q^b;Q^7)), EP, Q
import pickle, itertools, sympy as sp, time
from fractions import Fraction as Fr
P_=2147483629
NV=14  # J1..J10, K1..K3, EP
def c2(k): return k*(k-1)//2
def norm21(e):
    r=e%21; k=(e-r)//21
    co=(-1)**(k%2); qe=-21*c2(k)-r*k
    if r>10: r=21-r
    if r==0: return None
    return co,qe,r-1
def norm7(e):
    r=e%7; k=(e-r)//7
    co=(-1)**(k%2); qe=-7*c2(k)-r*k
    if r>3: r=7-r
    if r==0: return None
    return co,qe,10+r-1
def prod(es,nf):
    co=1; qe=0; J=[0]*NV
    for e in es:
        t=nf(e)
        if t is None: return None
        co*=t[0]; qe+=t[1]; J[t[2]]+=1
    return co,qe,tuple(J)
def inst(x,y,u,v,nf):
    a=prod([x+y,x-y,u+v,u-v],nf); b=prod([x+v,x-v,u+y,u-y],nf); c=prod([y+v,y-v,x+u,x-u],nf)
    poly={}
    for t,s,qs in ((a,1,0),(b,-1,0),(c,-1,u-y)):
        if t is None: continue
        key=(t[2],t[1]+qs); poly[key]=poly.get(key,0)+s*t[0]
    return {k:v for k,v in poly.items() if v}
def normalize(p):
    if not p: return None
    mq=min(k[1] for k in p); q={(k[0],k[1]-mq):v for k,v in p.items()}
    ks=sorted(q); s=1 if q[ks[0]]>0 else -1
    return tuple(sorted((k,s*v) for k,v in q.items()))
GENS={}
for x,y in itertools.product(range(-21,22),repeat=2):
    for u in range(1,11):
        for v in range(u+1,11):
            if 2*v<21:
                n=normalize(inst(x,y,u,v,norm21))
                if n and len(n)>1 and n not in GENS: GENS[n]=('W21',x,y,u,v)
for x,y in itertools.product(range(-7,8),repeat=2):
    for u,v in ((1,2),(1,3),(2,3)):
        n=normalize(inst(x,y,u,v,norm7))
        if n and len(n)>1 and n not in GENS: GENS[n]=('W7',x,y,u,v)
# product relations K_b EP^3 = J7 J_b J_{b+7} J_{7-b}
for b in (1,2,3):
    J1=[0]*NV; J1[10+b-1]=1; J1[13]=3
    J2=[0]*NV
    for a in (7,b,b+7,7-b): J2[a-1]+=1
    n=normalize({(tuple(J1),0):1,(tuple(J2),0):-1})
    GENS[n]=('P',b)
print("gens",len(GENS), sum(1 for v in GENS.values() if v[0]=='W7'))
pickle.dump(GENS,open('mixed_gens.pkl','wb'))
