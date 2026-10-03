import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; X=F.design(d)
tc=te.dropna(subset=list(F.NUMS.values())); Xt=F.design(tc); Xi=X
rng=np.random.default_rng(0); P=[];Pi=[]
for i in range(30):
    idx=rng.integers(0,len(d),len(d)); m=Ridge(1).fit(X[idx],y[idx]); P.append(m.predict(Xt)); Pi.append(m.predict(Xi))
P=np.array(P); Pi=np.array(Pi)
sd=P.std(0); sdi=Pi.std(0)
print('bootstrap pred sd: test complete rows mean var',np.mean(sd**2).round(3),'rms sd',np.sqrt(np.mean(sd**2)).round(3),'| train rows',np.sqrt(np.mean(sdi**2)).round(3))
print('test sd quantiles',np.quantile(sd,[.5,.9,.99]).round(2))
tc=tc.assign(sd=sd); print(tc.groupby('building_id').sd.apply(lambda s: np.sqrt((s**2).mean())).round(2).to_dict())
