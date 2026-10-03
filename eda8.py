from fw import *
import lightgbm as lgb
from sklearn.model_selection import cross_val_predict
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
X=d[['hour','dow','month','temperature','humidity','occupancy']].copy(); X['b']=d.building_id.astype('category')
S=cross_val_predict(lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=15,verbose=-1),X,y,cv=5)
d['S']=S; d['eS']=y-S; d['r']=d.previous_usage/y; d['dp']=d.previous_usage-y
g=d.groupby(pd.qcut(d.S,8))
print(g.agg(eS_std=('eS','std'),dp_std=('dp','std'),r_mean=('r','mean'),r_std=('r','std'),S=('S','mean')).round(3))
g=d.groupby('building_id'); print(g.agg(eS_std=('eS','std'),r_mean=('r','mean'),r_std=('r','std'),dp=('dp','mean'),S=('S','mean')).round(3))
g=d.groupby('hour'); print(g.agg(r_mean=('r','mean'),r_std=('r','std'),dp=('dp','mean')).round(3).T.to_string())
