import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, features as F, pandas as pd, numpy as np, joblib, base64
art=joblib.load('art_full_v6.pkl'); te=pd.read_csv('test.csv')
p=A.predict(art,te)
tc=F.clean(te); nm=tc[list(F.NUMS.values())].isna().sum(1).values; C=(tc.occupancy.isna()&tc.previous_usage.isna()).values
t=art['tree'].predict(F.gbm_frame(tc,F.FULL))
m=(nm>=2)&~C
p9=p.copy(); p9[m]=0.5*p[m]+0.5*t[m]
print('rows changed',m.sum(),'rms change',np.sqrt(np.mean((p9-p)[m]**2)))
d=pd.DataFrame({'id':te.id,'prediction':np.round(p9,2)}); d.to_csv('sub_v9.csv',index=False)
d['n']=d.id.str[2:].astype(int); d=d.sort_values('n'); v=np.round(d.prediction.values*100).astype('<u2'); print('sum',int(v.astype(np.int64).sum()))
# patch relative to v8
v8=pd.read_csv('sub_v8.csv'); v8['n']=v8.id.str[2:].astype(int); v8=v8.sort_values('n'); b8=np.round(v8.prediction.values*100).astype(int)
idx=np.where(v.astype(int)!=b8)[0]; arr=np.zeros(len(idx)*2,dtype='<u2'); arr[0::2]=idx; arr[1::2]=v[idx]
open('p9.b64','w').write(base64.b64encode(arr.tobytes()).decode()); print('patch rows',len(idx))
