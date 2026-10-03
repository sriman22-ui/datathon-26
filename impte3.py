import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
GP=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
def rdes(df,av):
    B=F._oh(df.building_id,F.BLD);H=F._oh(df.hour,range(24));W=F._oh(df.dow,range(7));M=F._oh(df.month,range(1,13))
    x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1); we=(df.dow.values>=5).astype(float)[:,None]
    parts=[B,H,W,M,x(H,B),x(we,B),x(H*we,F._oh(df.building_type,F.TYP))]
    if av: num=np.column_stack([df[F.NUMS[a]].values for a in av]); parts+=[num,x(num,B),x(num,F._oh(df.hour//4,range(6)))]
    return np.hstack(parts)
def xgX(df,av):
    X=F.gbm_frame(df,av).copy(); X['b']=X.b.cat.codes; return X
for tgt,av in [('o',('t','h','p')),('o',('t','h')),('p',('t','h','o')),('p',('t','h')),('o',('t','h','p')),]:
    cols=[F.NUMS[tgt]]+[F.NUMS[a] for a in av]
    a=tr.dropna(subset=cols); b=te.dropna(subset=cols).reset_index(drop=True); yb=b[F.NUMS[tgt]].values
    P={k:np.zeros(len(b)) for k in ['lgb','xgb','ridge']}
    for x,v in KFold(5,shuffle=True,random_state=0).split(b):
        ab=pd.concat([a,b.iloc[x]]); w=np.r_[np.ones(len(a)),10*np.ones(len(x))]; yy=ab[F.NUMS[tgt]].values
        P['lgb'][v]=lgb.LGBMRegressor(**GP).fit(F.gbm_frame(ab,av),yy,sample_weight=w).predict(F.gbm_frame(b.iloc[v],av))
        P['xgb'][v]=xgb.XGBRegressor(n_estimators=500,learning_rate=0.04,max_depth=6,subsample=0.8,colsample_bytree=0.9,n_jobs=2).fit(xgX(ab,av),yy,sample_weight=w).predict(xgX(b.iloc[v],av))
        P['ridge'][v]=Ridge(3).fit(rdes(ab,av),yy,sample_weight=w).predict(rdes(b.iloc[v],av))
    r=lambda p: round(np.sqrt(np.mean((p-yb)**2)),2)
    print(tgt,av,{k:r(p) for k,p in P.items()},'lgb+xgb',r((P['lgb']+P['xgb'])/2),'all3',r((P['lgb']+P['xgb']+P['ridge'])/3),'.4l.4x.2r',r(.4*P['lgb']+.4*P['xgb']+.2*P['ridge']),flush=True)
    if tgt=='o' and len(av)==3: break
