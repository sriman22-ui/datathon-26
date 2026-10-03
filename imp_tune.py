import sys; sys.path.insert(0,'ml_pipeline')
from model_lib import *
tr=pd.read_csv('ml_pipeline/split_train.csv'); va=pd.read_csv('ml_pipeline/split_val.csv')
d=pd.concat([tr,va]).reset_index(drop=True).dropna().reset_index(drop=True)
kf=list(KFold(5,shuffle=True,random_state=0).split(d))
for tgt in ['o','p']:
    av=tuple(a for a in F.FULL if a!=tgt); yv=d[F.NUMS[tgt]].values
    for name,gp in [('base',{}),('slow15',dict(n_estimators=1000,learning_rate=0.02,num_leaves=15)),('slow31',dict(n_estimators=800,learning_rate=0.02,num_leaves=31,min_child_samples=30)),('l63',dict(n_estimators=500,learning_rate=0.03,num_leaves=63))]:
        g=np.zeros(len(d))
        for a,b in kf: g[b]=lgb.LGBMRegressor(**{**GP,'n_jobs':2,**gp}).fit(F.gbm_frame(d.iloc[a],av),yv[a]).predict(F.gbm_frame(d.iloc[b],av))
        print(tgt,name,round(rmse(g,yv),3),flush=True)
