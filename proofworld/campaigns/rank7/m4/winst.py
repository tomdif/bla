# generate Weierstrass instances as polynomials: dict {(Jexps tuple(10), Qexp): coeff}
import itertools, pickle
from fractions import Fraction as Fr
def c2(k): return k*(k-1)//2
def thn(e):
    r=e%21; k=(e-r)//21
    co=(-1)**(k%2); qe=-21*c2(k)-r*k
    if r>10: r=21-r
    if r==0: return None
    return co,qe,r
def prod(es):
    co=1; qe=0; J=[0]*10
    for e in es:
        t=thn(e)
        if t is None: return None
        co*=t[0]; qe+=t[1]; J[t[2]-1]+=1
    return co,qe,tuple(J)
def inst(x2,y2,u2,v2):  # doubled coordinates
    terms=[]
    a=prod([(x2+y2)//2,(x2-y2)//2,(u2+v2)//2,(u2-v2)//2])
    b=prod([(x2+v2)//2,(x2-v2)//2,(u2+y2)//2,(u2-y2)//2])
    c=prod([(y2+v2)//2,(y2-v2)//2,(x2+u2)//2,(x2-u2)//2])
    poly={}
    for t,s,qs in ((a,1,0),(b,-1,0),(c,-1,(u2-y2)//2)):
        if t is None: continue
        key=(t[2],t[1]+qs); poly[key]=poly.get(key,0)+s*t[0]
    poly={k:v for k,v in poly.items() if v}
    return poly
def normalize(poly):
    # canonical: shift Q so min exponent 0, sign so first key coeff positive
    if not poly: return None
    mq=min(k[1] for k in poly)
    p={(k[0],k[1]-mq):v for k,v in poly.items()}
    ks=sorted(p); s=1 if p[ks[0]]>0 else -1
    return tuple(sorted((k,s*v) for k,v in p.items()))
seen=set()
for par in (0,1):
    rng=range(par,42,2)
    for x2,y2,u2,v2 in itertools.product(rng,repeat=4):
        p=inst(x2,y2,u2,v2)
        n=normalize(p)
        if n and len(n)>1: seen.add(n)
if __name__=="__main__": print(len(seen))
if __name__=="__main__": pickle.dump(list(seen),open('winst.pkl','wb'))
