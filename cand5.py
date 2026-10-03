import sys; sys.path.insert(0,'ml_pipeline')
import artifact as A, pandas as pd, numpy as np, joblib
dev=pd.concat([pd.read_csv('ml_pipeline/split_train.csv'),pd.read_csv('ml_pipeline/split_val.csv')]); dev['id']='x'
te=pd.read_csv('test.csv')
art=A.build(dev,unlabeled=te); joblib.dump(art,'art_dev_v3.pkl')
l=pd.read_csv('sub_v0_lgb.csv').prediction.values; v3=pd.read_csv('sub_v3.csv').prediction.values; r2=pd.read_csv('sub_v2_ridge_dev.csv').prediction.values
C=(te.occupancy.isna()&te.previous_usage.isna()).values; M=te[['temperature','humidity','occupancy','previous_usage']].isna().any(axis=1).values
pr=A.predict(art,te,c_mode='ridge')
print('C rows: new ridge-imp vs LGB rms',np.sqrt(np.mean((pr[C]-l[C])**2)),'old ridge vs LGB',np.sqrt(np.mean((r2[C]-l[C])**2)),'mean new-old',(pr[C]-r2[C]).mean())
print('missing rows: mean change new-old ridge',(pr[M&~C]-r2[M&~C]).mean())
for cm in ['tree','blend','ridge']:
    p=A.predict(art,te,c_mode=cm); pd.DataFrame({'id':te.id,'prediction':np.round(p,2)}).to_csv(f'sub_v5_{cm}.csv',index=False)
