import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np, joblib
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
art=A.build(tr,unlabeled=te,c_mode='blend')
joblib.dump(art,'deliver/model.pkl',compress=3)
p=A.predict(art,te); pd.DataFrame({'id':te.id,'prediction':p}).to_csv('deliver/submission.csv',index=False)
print(pd.Series(p).describe())
