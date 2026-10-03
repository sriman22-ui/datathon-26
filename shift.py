import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
GP=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
for tgt in ['o','p','t','h']:
    for av in [tuple(a for a in F.FULL if a!=tgt), ('t','h') if tgt in 'op' else ()]:
        cols=[F.NUMS[tgt]]+[F.NUMS[a] for a in av]
        a=tr.dropna(subset=cols); b=te.dropna(subset=cols)
        g=lgb.LGBMRegressor(**GP).fit(F.gbm_frame(a,av),a[F.NUMS[tgt]]); pr=g.predict(F.gbm_frame(b,av))
        res=b[F.NUMS[tgt]].values-pr
        print(tgt,'given',av,'test bias',round(res.mean(),2),'rmse',round(np.sqrt((res**2).mean()),2),'| train-OOF-ish sd', round(a[F.NUMS[tgt]].std(),1))
