import pandas as pd, numpy as np
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
tr['dow']=tr.day_of_week.map({d:i for i,d in enumerate(D)})
d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
def lin(X):
    p=cross_val_predict(RidgeCV(alphas=np.logspace(-4,2,20)),X,y,cv=KFold(5,shuffle=True,random_state=0)); return np.sqrt(np.mean((p-y)**2)),p
B=pd.get_dummies(d.building_id,dtype=float); H=pd.get_dummies(d.hour,prefix='h',dtype=float); W=pd.get_dummies(d.dow,prefix='w',dtype=float); M=pd.get_dummies(d.month,prefix='m',dtype=float)
num=d[['temperature','humidity','occupancy','previous_usage']]
X1=pd.concat([B,H,W,M,num],axis=1); print('additive',lin(X1)[0])
# building interactions with num
inter=pd.concat([B.mul(d[c],axis=0).add_suffix('_'+c) for c in num],axis=1)
X2=pd.concat([X1,inter],axis=1); print('+B x num',lin(X2)[0])
BH=pd.concat([B.mul(H[h],axis=0).add_suffix('_'+h) for h in H],axis=1)
X3=pd.concat([X2,BH],axis=1); print('+B x hour',lin(X3)[0])
BW=pd.concat([B.mul((d.dow>=5).astype(float),axis=0).add_suffix('_we')],axis=1)
X4=pd.concat([X3,BW,pd.concat([B.mul(M[m],axis=0).add_suffix(m) for m in M],axis=1)],axis=1); print('+B x weekend, B x month',lin(X4)[0])
r,p=lin(X4); res=y-p
d['res']=res
for c in ['hour','dow','month','building_id']: print(d.groupby(c).res.agg(['mean','std']).round(2).T.to_string())
for c in ['temperature','humidity','occupancy','previous_usage']:
    print(c, d.groupby(pd.qcut(d[c],10,duplicates='drop')).res.agg(['mean','std']).round(2).T.to_string())
