import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
wr=lambda p: np.sqrt(np.sum(w*(p-y)**2)/w.sum()); r=lambda p: np.sqrt(np.mean((p-y)**2))
P0=np.zeros(len(d)); P1=np.zeros(len(d)); P2=np.zeros(len(d))
for a,b in KFold(5,shuffle=True,random_state=0).split(d):
    A=d.iloc[a]; ra=np.zeros(len(a))
    for a2,b2 in KFold(5,shuffle=True,random_state=1).split(A):
        ra[b2]=y[a][b2]-Ridge(1).fit(F.design(A.iloc[a2]),y[a][a2]).predict(F.design(A.iloc[b2]))
    m=Ridge(1).fit(F.design(A),y[a]); P0[b]=m.predict(F.design(d.iloc[b]))
    X=F.gbm_frame(A,F.FULL); X['ridge']=y[a]-ra; Xb=F.gbm_frame(d.iloc[b],F.FULL); Xb['ridge']=P0[b]
    g=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.02,num_leaves=7,min_child_samples=50,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2).fit(X,ra)
    P1[b]=P0[b]+g.predict(Xb)
    g=lgb.LGBMRegressor(n_estimators=300,learning_rate=0.02,num_leaves=7,min_child_samples=50,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2).fit(X,ra,sample_weight=w[a])
    P2[b]=P0[b]+g.predict(Xb)
for n,p in [('ridge',P0),('ridge+resid',P1),('ridge+resid_w',P2)]: print(n,round(r(p),4),round(wr(p),4))
