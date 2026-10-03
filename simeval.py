from pipe import *
import sys
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
rng=np.random.default_rng(0)
tm=te[[CMAP[a] for a in FULL]].isna().values
masks=tm[rng.integers(0,len(tm),len(d))]  # sample test patterns
def run(mk):
    res={}
    for name,fl in folds(d):
        P=np.full(len(y),np.nan)
        for a,b in fl:
            m=mk().fit(d.iloc[a],y[a]); dt=d.iloc[b].copy()
            for j,c in enumerate([CMAP[x] for x in FULL]): dt.loc[masks[b,j],c]=np.nan
            P[b]=m.predict(dt)
        k=~np.isnan(P); res[name]=rmse(P[k],y[k])
    print('  '.join(f'{k}={v:.4f}' for k,v in res.items()))
for mode in ['direct','imp','blend']:
    print(mode,end=': '); run(lambda: Model(mode=mode))
