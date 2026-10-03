import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_predict
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
tr['dow']=tr.day_of_week.map({d:i for i,d in enumerate(D)})
y=tr.energy_usage.values
def cvp(cols,data=tr,target=None):
    X=data[cols].copy()
    for c in X: 
        if X[c].dtype==object: X[c]=X[c].astype('category')
    t=data.energy_usage.values if target is None else target
    m=lgb.LGBMRegressor(n_estimators=800,learning_rate=0.03,num_leaves=15,verbose=-1)
    return cross_val_predict(m,X,t,cv=KFold(5,shuffle=True,random_state=0))
base=['building_id','hour','dow','month','temperature','humidity','occupancy']
p0=cvp(base); print('no prev rmse',np.sqrt(np.mean((p0-y)**2)))
# predict prev from base too
d=tr[tr.previous_usage.notna()].reset_index(drop=True)
pp=cvp(base,d,d.previous_usage.values); print('prev from base rmse',np.sqrt(np.mean((pp-d.previous_usage)**2)))
pe=cvp(base,d); 
re=d.energy_usage-pe; rp=d.previous_usage-pp
print('corr of residuals',np.corrcoef(re,rp)[0,1], re.std(), rp.std())
print(np.polyfit(rp,re,1))
