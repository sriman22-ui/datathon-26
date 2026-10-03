import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
base=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
cfgs={'w3':(3,base),'w10':(10,base),'w3_slow':(3,dict(base,n_estimators=800,learning_rate=0.02,num_leaves=15)),'w5_l63':(5,dict(base,num_leaves=63,min_child_samples=10))}
for tgt,av in [('o',('t','h','p')),('o',('t','h')),('p',('t','h','o')),('p',('t','h'))]:
    cols=[F.NUMS[tgt]]+[F.NUMS[a] for a in av]
    a=tr.dropna(subset=cols); b=te.dropna(subset=cols).reset_index(drop=True); yb=b[F.NUMS[tgt]].values
    P={k:np.zeros(len(b)) for k in cfgs}
    for x,v in KFold(5,shuffle=True,random_state=0).split(b):
        ab=pd.concat([a,b.iloc[x]])
        for k,(w,gp) in cfgs.items():
            P[k][v]=lgb.LGBMRegressor(**gp).fit(F.gbm_frame(ab,av),ab[F.NUMS[tgt]],sample_weight=np.r_[np.ones(len(a)),w*np.ones(len(x))]).predict(F.gbm_frame(b.iloc[v],av))
    print(tgt,av,{k:round(np.sqrt(np.mean((p-yb)**2)),2) for k,p in P.items()},flush=True)
