import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np
dev=pd.concat([pd.read_csv('ml_pipeline/split_train.csv'),pd.read_csv('ml_pipeline/split_val.csv')])
dev['id']='x'
art=A.build(dev); te=pd.read_csv('test.csv'); p=A.predict(art,te)
pd.DataFrame({'id':te.id,'prediction':p}).to_csv('sub_v2_ridge_dev.csv',index=False)
print(pd.Series(p).describe()); print(np.isnan(p).sum())
old=pd.read_csv('sub_v1_ridge_imp.csv'); print('corr with v1',np.corrcoef(old.prediction,p)[0,1], np.sqrt(np.mean((old.prediction-p)**2)))
