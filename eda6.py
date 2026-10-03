import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import Ridge
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
tr['dow']=tr.day_of_week.map({d:i for i,d in enumerate(D)})
d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
B=pd.get_dummies(d.building_id,dtype=float); H=pd.get_dummies(d.hour,prefix='h',dtype=float); W=pd.get_dummies(d.dow,prefix='w',dtype=float); M=pd.get_dummies(d.month,prefix='m',dtype=float)
def bx(cols): return pd.concat([B.mul(cols[c],axis=0).add_suffix('_'+c) for c in cols],axis=1)
# temperature as bins (one-hot, 0.5 deg), per type scale? first global bins
tb=pd.get_dummies(np.clip((d.temperature*2).round()/2,24,33.5),prefix='T',dtype=float)
num=pd.DataFrame({'hu':d.humidity,'o':d.occupancy,'p':d.previous_usage})
X=pd.concat([B,H,W,M,num,bx(num),bx(H),bx(pd.DataFrame({'we':(d.dow>=5)*1.0})),tb],axis=1)
m=Ridge(1e-3).fit(X,y)
co=pd.Series(m.coef_,X.columns); print(co[[c for c in X.columns if c.startswith('T_')]].round(2).to_string())
# occupancy bins
ob=pd.get_dummies(pd.cut(d.occupancy,[-1,0,5,10,20,40,60,80,100,130,160,200,250,400]),prefix='O',dtype=float)
X=pd.concat([B,H,W,M,d[['temperature','humidity','previous_usage']],bx(d[['temperature']]),bx(H),ob],axis=1)
m=Ridge(1e-3).fit(X,y); co=pd.Series(m.coef_,X.columns); print(co[[c for c in X.columns if c.startswith('O_')]].round(2).to_string())
# humidity bins
hb=pd.get_dummies(pd.cut(d.humidity,[0,66,70,74,78,82,86,90,101]),prefix='HU',dtype=float)
X=pd.concat([B,H,W,M,num,bx(num),bx(d[['temperature']]),bx(H),hb],axis=1)
m=Ridge(1e-3).fit(X,y); co=pd.Series(m.coef_,X.columns); print(co[[c for c in X.columns if c.startswith('HU_')]].round(2).to_string(), co['hu'])
