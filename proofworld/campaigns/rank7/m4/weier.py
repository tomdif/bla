from num import th
import cmath, random
Qv=0.6*cmath.exp(0.3j); P=Qv**21
T=lambda e: th(Qv**e,P)
for _ in range(5):
    x,y,u,v=[random.randint(-30,30)/2 for _ in range(4)]
    if random.random()<.5: x,y,u,v=[round(a) for a in (x,y,u,v)]
    else: x,y,u,v=[round(a)+0.5 for a in (x,y,u,v)]
    val=T(x+y)*T(x-y)*T(u+v)*T(u-v)-T(x+v)*T(x-v)*T(u+y)*T(u-y)-Qv**(u-y)*T(y+v)*T(y-v)*T(x+u)*T(x-u)
    print(x,y,u,v,abs(val))
