from fw import *
import lightgbm as lgb
from sklearn.linear_model import Ridge, LinearRegression
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
def gbX(df,cols):
    X=df[cols].copy(); 
    if 'building_id' in cols: X['building_id']=pd.Categorical(X.building_id,BLD)
    return X
P=dict(n_estimators=1500,learning_rate=0.02,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
allc=['building_id','hour','dow','month','temperature','humidity','occupancy','previous_usage']
def m_lgb(dtr,ytr,dte,**kw):
    m=lgb.LGBMRegressor(**{**P,**kw}).fit(gbX(dtr,allc),ytr); return m.predict(gbX(dte,allc))
def m_off(a,cols,**kw):
    def f(dtr,ytr,dte):
        m=lgb.LGBMRegressor(**{**P,**kw}).fit(gbX(dtr,cols),ytr-a*dtr.previous_usage.values); return m.predict(gbX(dte,cols))+a*dte.previous_usage.values
    return f
print('lgb plain',end=': '); evaluate(m_lgb,d,y)
print('lgb lineartree',end=': '); evaluate(lambda a,b,c: m_lgb(a,b,c,linear_tree=True,linear_lambda=1.0),d,y)
nop=[c for c in allc if c!='previous_usage']
for a in [0.3,0.35,0.4]:
    print('offset',a,end=': '); evaluate(m_off(a,nop),d,y)
print('offset .35 + prev',end=': '); evaluate(m_off(0.35,allc),d,y)
