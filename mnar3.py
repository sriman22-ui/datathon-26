import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
tr=pd.read_csv('train_oof_imp.csv'); trc=F.clean(tr); y=tr.energy_usage.values
gp=dict(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2)
L=np.zeros(len(tr))
for s in range(2):
  for a,b in KFold(5,shuffle=True,random_state=s).split(tr):
    L[b]+=lgb.LGBMRegressor(**gp,random_state=s).fit(F.gbm_frame(trc.iloc[a],F.FULL),y[a]).predict(F.gbm_frame(trc.iloc[b],F.FULL))/2
tr['lgb']=L
for c in F.NUMS.values():
    m=tr[c].isna()
    print(c.ljust(15),'n',m.sum(),'imp rmse',round(np.sqrt(((tr.oof-y)[m]**2).mean()),3),'lgb rmse',round(np.sqrt(((tr.lgb-y)[m]**2).mean()),3),'blend',round(np.sqrt(((0.5*tr.oof+0.5*tr.lgb-y)[m]**2).mean()),3))
tr.to_csv('train_oof_imp.csv',index=False)
