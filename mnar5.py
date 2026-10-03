import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
GP=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
N=F.NUMS
for name,df in [('train',tr),('test',te)]:
    print('==',name)
    for obs in ['o','p','t','h']:
        # residual of observed var given cats only (OOF within df), compare rows where another var is missing
        d=df.dropna(subset=[N[obs]]).reset_index(drop=True); y=d[N[obs]].values; r=np.zeros(len(d))
        for a,b in KFold(5,shuffle=True,random_state=0).split(d):
            r[b]=y[b]-lgb.LGBMRegressor(**GP).fit(F.gbm_frame(d.iloc[a],()),y[a]).predict(F.gbm_frame(d.iloc[b],()))
        s=r.std()
        out=[]
        for mis in ['o','p','t','h']:
            if mis==obs: continue
            m=d[N[mis]].isna().values
            out.append(f"{mis}-missing: n={m.sum()} resid mean={r[m].mean():+.2f} ({r[m].mean()/s*np.sqrt(m.sum()):+.1f} se)")
        print(f"observed {obs} (sd {s:.1f}):", ' | '.join(out))
