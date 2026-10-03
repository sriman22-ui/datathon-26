import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
P={k:np.zeros(len(d)) for k in ['ridge','ridge_w','lgb','lgb_w']}
for a,b in KFold(5,shuffle=True,random_state=0).split(d):
    A,Bv=d.iloc[a],d.iloc[b]
    P['ridge'][b]=Ridge(1).fit(F.design(A),y[a]).predict(F.design(Bv))
    P['ridge_w'][b]=Ridge(1).fit(F.design(A),y[a],sample_weight=w[a]).predict(F.design(Bv))
    gp=dict(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2)
    P['lgb'][b]=lgb.LGBMRegressor(**gp).fit(F.gbm_frame(A,F.FULL),y[a]).predict(F.gbm_frame(Bv,F.FULL))
    P['lgb_w'][b]=lgb.LGBMRegressor(**gp).fit(F.gbm_frame(A,F.FULL),y[a],sample_weight=w[a]).predict(F.gbm_frame(Bv,F.FULL))
wr=lambda p: np.sqrt(np.sum(w*(p-y)**2)/w.sum()); r=lambda p: np.sqrt(np.mean((p-y)**2))
for k,p in P.items(): print(k.ljust(8),'rmse',round(r(p),4),'adv-weighted rmse',round(wr(p),4))
for a in [0.2,0.4]: p=(1-a)*P['ridge']+a*P['lgb']; print('blend',a,round(r(p),4),round(wr(p),4))
np.save('wcv_oof.npy',np.vstack([P[k] for k in P]))
