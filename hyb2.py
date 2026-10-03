from ridgelib import *
import lightgbm as lgb
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
print('ridge',end=': '); evaluate(ridge_fp(),d,y)
P=dict(n_estimators=400,learning_rate=0.02,num_leaves=7,min_child_samples=40,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,reg_lambda=5,verbose=-1)
def hyb(cols,**kw):
    rf=ridge_fp()
    def f(dtr,ytr,dte):
        # inner oof ridge residuals to avoid overfit residuals
        from sklearn.model_selection import KFold
        r=np.zeros(len(ytr))
        for a,b in KFold(5,shuffle=True,random_state=1).split(dtr): r[b]=ytr[b]-rf(dtr.iloc[a],ytr[a],dtr.iloc[b])
        base=rf(dtr,ytr,dte)
        X=dtr[cols].copy(); Xt=dte[cols].copy()
        if 'building_id' in cols: X['building_id']=pd.Categorical(X.building_id,BLD); Xt['building_id']=pd.Categorical(Xt.building_id,BLD)
        m=lgb.LGBMRegressor(**{**P,**kw}).fit(X,r); return base+m.predict(Xt)
    return f
print('hyb cat',end=': '); evaluate(hyb(['building_id','hour','dow','month']),d,y)
print('hyb all',end=': '); evaluate(hyb(['building_id','hour','dow','month','temperature','humidity','occupancy','previous_usage']),d,y)
