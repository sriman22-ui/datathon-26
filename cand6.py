import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np, joblib
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
art=A.build(tr,unlabeled=te,c_mode='blend'); joblib.dump(art,'art_full_v4.pkl',compress=3)
p=A.predict(art,te); pd.DataFrame({'id':te.id,'prediction':np.round(p,2)}).to_csv('sub_v6.csv',index=False)
v5=pd.read_csv('sub_v5_blend.csv').prediction.values
M=te[['temperature','humidity','occupancy','previous_usage']].isna().any(axis=1).values
print('rms diff vs v5 complete',np.sqrt(np.mean((p[~M]-v5[~M])**2)),'missing',np.sqrt(np.mean((p[M]-v5[M])**2)))
