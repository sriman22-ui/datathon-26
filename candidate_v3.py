import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np, joblib
dev=pd.concat([pd.read_csv('ml_pipeline/split_train.csv'),pd.read_csv('ml_pipeline/split_val.csv')]); dev['id']='x'
art=A.build(dev); joblib.dump(art,'art_dev.pkl')
te=pd.read_csv('test.csv'); p=A.predict(art,te)
pd.DataFrame({'id':te.id,'prediction':np.round(p,2)}).to_csv('sub_v3.csv',index=False)
v2=pd.read_csv('sub_v2_ridge_dev.csv').prediction.values; l=pd.read_csv('sub_v0_lgb.csv').prediction.values
both=te.occupancy.isna()&te.previous_usage.isna()
print('rows via tree',both.sum(),'rms diff to v2',np.sqrt(np.mean((p-v2)**2)),'C-rows diff vs probe LGB',np.abs(p[both]-l[both]).max())
