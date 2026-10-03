import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
both=pd.concat([tr.assign(is_te=0),te.assign(is_te=1)]).reset_index(drop=True)
X=F.gbm_frame(both,F.FULL)
p=cross_val_predict(lgb.LGBMClassifier(n_estimators=300,learning_rate=0.03,num_leaves=15,verbose=-1,n_jobs=2),X,both.is_te,cv=5,method='predict_proba')[:,1]
print('adv AUC',roc_auc_score(both.is_te,p))
ptr=p[:len(tr)]; w=ptr/(1-ptr); w=w/w.mean()*1.0
print('weight quantiles',np.round(np.quantile(w,[.01,.25,.5,.75,.99,.999]),2),'ESS',w.sum()**2/(w**2).sum())
tr['w']=w; tr.to_csv('train_advw.csv',index=False)
d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True)
print(d.assign(tb=d.hour.isin([22,23,0,1,2,3,4,7,8])).groupby('tb').w.mean())
print('hot w',d[d.temperature>33.5].w.mean(),'hum w',d[d.humidity>93].w.mean())
