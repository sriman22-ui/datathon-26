import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
tr['dow']=tr.day_of_week.map({d:i for i,d in enumerate(D)})
d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
B=pd.get_dummies(d.building_id,dtype=float); H=pd.get_dummies(d.hour,prefix='h',dtype=float)
def bx(cols): return pd.concat([B.mul(cols[c],axis=0).add_suffix('_'+c) for c in cols],axis=1)
num=pd.DataFrame({'hu':d.humidity,'o':d.occupancy,'p':d.previous_usage})
base=[B,H,pd.get_dummies(d.dow,prefix='w',dtype=float),pd.get_dummies(d.month,prefix='m',dtype=float),num,bx(num),bx(H)]
for b in ['SCI_A','RES_A','ADM_A','ENG_A']:
  mask=(d.building_id==b).values
  tb=pd.get_dummies(np.clip(d.temperature.round(),24,34),prefix='T',dtype=float)
  X=pd.concat(base+[tb.mul(mask,axis=0)],axis=1)
  m=Ridge(1e-2).fit(X,y); co=pd.Series(m.coef_,X.columns)
  print(b, co[[c for c in X.columns if c.startswith('T_')]].round(1).values, d[mask].temperature.gt(32).sum())
