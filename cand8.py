import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np, joblib, os
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
art=A.build(tr,unlabeled=te,c_mode='blend'); joblib.dump(art,'art_full_v6.pkl',compress=('xz',9)); print('MB',os.path.getsize('art_full_v6.pkl')/1e6)
p=A.predict(art,te); pd.DataFrame({'id':te.id,'prediction':np.round(p,2)}).to_csv('sub_v8.csv',index=False)
v6=pd.read_csv('sub_v6.csv').prediction.values; M=te[['temperature','humidity','occupancy','previous_usage']].isna().any(axis=1).values
print('rms diff v8-v6 complete',np.sqrt(np.mean((p[~M]-v6[~M])**2)),'missing',np.sqrt(np.mean((p[M]-v6[M])**2)))
d=pd.DataFrame({'id':te.id,'prediction':np.round(p,2)}); d['n']=d.id.str[2:].astype(int); d=d.sort_values('n')
import base64; v=np.round(d.prediction.values*100).astype('<u2'); open('v8.b64','w').write(base64.b64encode(v.tobytes()).decode()); print('sum',int(v.astype(np.int64).sum()))
