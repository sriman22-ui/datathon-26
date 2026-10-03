import sys; sys.path.insert(0,'ml_pipeline')
import features as F, artifact as A, numpy as np, pandas as pd, lightgbm as lgb, joblib
art=joblib.load('art_dev.pkl')
dev=F.clean(pd.concat([pd.read_csv('ml_pipeline/split_train.csv'),pd.read_csv('ml_pipeline/split_val.csv')]))
te=F.clean(pd.read_csv('test.csv')); l=pd.read_csv('sub_v0_lgb.csv').prediction.values
C=(te.occupancy.isna()&te.previous_usage.isna()&te.temperature.notna()&te.humidity.notna()).values
x=te[C].copy(); av=('t','h'); L=l[C]
for q in [0.5,0.75,0.9,0.95]:
    s=x.copy()
    for mv in ['o','p']:
        sub=dev.dropna(subset=[F.NUMS[mv],'temperature','humidity'])
        g=lgb.LGBMRegressor(objective='quantile',alpha=q,n_estimators=400,learning_rate=0.04,num_leaves=31,verbose=-1,n_jobs=2).fit(F.gbm_frame(sub,av),sub[F.NUMS[mv]])
        s[F.NUMS[mv]]=g.predict(F.gbm_frame(x,av))
    p=art['main'].predict(F.design(s,F.FULL,art['hinge']))
    print('q',q,'mean(LGB - ridge@q)',round((L-p).mean(),2),'rms',round(np.sqrt(((L-p)**2).mean()),2),'corr',round(np.corrcoef(L,p)[0,1],3))
