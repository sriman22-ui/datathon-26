import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, features as F, numpy as np, pandas as pd
from sklearn.model_selection import KFold
tr=pd.read_csv('train.csv'); y=tr.energy_usage.values
oof=np.zeros(len(tr))
for a,b in KFold(5,shuffle=True,random_state=0).split(tr):
    art=A.build(tr.iloc[a]); oof[b]=A.predict(art,tr.iloc[b])
tr['oof']=oof; tr['res']=y-oof
for c in F.NUMS.values():
    m=tr[c].isna(); print(c,'missing n',m.sum(),'rmse',round(np.sqrt((tr.res[m]**2).mean()),3),'bias',round(tr.res[m].mean(),3))
m=tr[list(F.NUMS.values())].isna().any(axis=1); print('complete rmse',np.sqrt((tr.res[~m]**2).mean()))
tr.to_csv('train_oof_imp.csv',index=False)
