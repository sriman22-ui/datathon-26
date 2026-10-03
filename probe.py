import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
d=tr.dropna(subset=list(F.NUMS.values()))
av=('t','h','o')
m=Ridge(1).fit(F.design(d,av),d.energy_usage)
for n,df in [('tr',tr),('te',te)]:
    ok=df[['temperature','humidity','occupancy','previous_usage']].notna().all(1)
    x=df[ok]; S=m.predict(F.design(x,av)); r=x.previous_usage/S
    print(n,len(x),'ratio quantiles',np.round(np.quantile(r,[.001,.01,.05,.25,.5,.75,.95,.99,.999]),2), 'frac<0.6',round((r<0.6).mean(),4),'frac>1.4',round((r>1.4).mean(),4))
    if n=='te':
        x=x.assign(S=S,r=r); print(x[x.r>1.4].head(15)[['building_id','hour','day_of_week','temperature','humidity','occupancy','previous_usage','S','r']])
        print(x[x.r<0.6].head(10)[['building_id','hour','temperature','humidity','occupancy','previous_usage','S','r']])
    if n=='tr':
        x=x.assign(S=S,r=r); print(x[x.r>1.4][['building_id','hour','occupancy','previous_usage','S','r','energy_usage']].head(15))
