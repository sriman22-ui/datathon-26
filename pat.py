from ridgelib import *
import itertools
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
print(d[NUM].corr().round(2))
full=('t','h','o','p')
for k in range(0,4):
    for miss in itertools.combinations(full,k):
        av=tuple(c for c in full if c not in miss)
        r,_=evaluate(ridge_fp(avail=av),d,y,verbose=False)
        print('missing',miss,{k:round(v,3) for k,v in r.items() if k in('kf','ext_temp','ext_occu')})
