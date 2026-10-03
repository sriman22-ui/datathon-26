import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np, joblib, os
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
art=A.build(tr,unlabeled=te); joblib.dump(art,'art_full_v5.pkl',compress=('xz',9)); print('size MB',os.path.getsize('art_full_v5.pkl')/1e6)
print({''.join(k):round(v,2) for k,v in art['offsets'].items()})
p=A.predict(art,te); pd.DataFrame({'id':te.id,'prediction':np.round(p,2)}).to_csv('sub_v7.csv',index=False)
v6=pd.read_csv('sub_v6.csv').prediction.values; M=te[['temperature','humidity','occupancy','previous_usage']].isna().any(axis=1).values
print('rms diff v7-v6 complete',np.sqrt(np.mean((p[~M]-v6[~M])**2)),'missing',np.sqrt(np.mean((p[M]-v6[M])**2)),'mean diff missing',(p[M]-v6[M]).mean())
