import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
GP=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
for tgt,av in [('o',('t','h','p')),('o',('t','h')),('p',('t','h','o')),('p',('t','h')),('t',('h','o','p')),('h',('t','o','p'))]:
    cols=[F.NUMS[tgt]]+[F.NUMS[a] for a in av]
    a=tr.dropna(subset=cols); b=te.dropna(subset=cols).reset_index(drop=True); yb=b[F.NUMS[tgt]].values
    P={'train':np.zeros(len(b)),'both':np.zeros(len(b)),'both_w3':np.zeros(len(b)),'test':np.zeros(len(b))}
    for i,(x,v) in enumerate(KFold(5,shuffle=True,random_state=0).split(b)):
        bt=b.iloc[x]
        if i==0: g0=lgb.LGBMRegressor(**GP).fit(F.gbm_frame(a,av),a[F.NUMS[tgt]])
        P['train'][v]=g0.predict(F.gbm_frame(b.iloc[v],av))
        ab=pd.concat([a,bt]); w=np.r_[np.ones(len(a)),np.ones(len(bt))]
        P['both'][v]=lgb.LGBMRegressor(**GP).fit(F.gbm_frame(ab,av),ab[F.NUMS[tgt]]).predict(F.gbm_frame(b.iloc[v],av))
        P['both_w3'][v]=lgb.LGBMRegressor(**GP).fit(F.gbm_frame(ab,av),ab[F.NUMS[tgt]],sample_weight=np.r_[np.ones(len(a)),3*np.ones(len(bt))]).predict(F.gbm_frame(b.iloc[v],av))
        P['test'][v]=lgb.LGBMRegressor(**GP).fit(F.gbm_frame(bt,av),bt[F.NUMS[tgt]]).predict(F.gbm_frame(b.iloc[v],av))
    print(tgt,av,{k:(round(np.sqrt(np.mean((p-yb)**2)),2),round(np.mean(yb-p),2)) for k,p in P.items()},flush=True)
