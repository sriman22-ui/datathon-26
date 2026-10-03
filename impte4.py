import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb, json
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
def xgX(df,av): X=F.gbm_frame(df,av).copy(); X['b']=X.b.cat.codes; return X
L=lambda **k: dict(dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2),**k)
cands={
 'lgb_w10':('l',10,L()),'lgb_w30':('l',30,L()),'lgb_w10_l15':('l',10,L(num_leaves=15,n_estimators=700,learning_rate=0.03)),
 'lgb_w10_mcs40':('l',10,L(min_child_samples=40)),'xgb_w10':('x',10,dict(max_depth=6)),'xgb_w10_d4':('x',10,dict(max_depth=4,n_estimators=800)),'xgb_w30':('x',30,dict(max_depth=6)),
}
out={}
for tgt,av in [('o',('t','h','p')),('o',('t','h')),('p',('t','h','o')),('p',('t','h'))]:
    cols=[F.NUMS[tgt]]+[F.NUMS[a] for a in av]
    a=tr.dropna(subset=cols); b=te.dropna(subset=cols).reset_index(drop=True); yb=b[F.NUMS[tgt]].values
    P={k:np.zeros(len(b)) for k in cands}
    for x,v in KFold(5,shuffle=True,random_state=0).split(b):
        ab=pd.concat([a,b.iloc[x]]); yy=ab[F.NUMS[tgt]].values
        for k,(kind,w,gp) in cands.items():
            sw=np.r_[np.ones(len(a)),w*np.ones(len(x))]
            if kind=='l': P[k][v]=lgb.LGBMRegressor(**gp).fit(F.gbm_frame(ab,av),yy,sample_weight=sw).predict(F.gbm_frame(b.iloc[v],av))
            else:
                g=dict(n_estimators=500,learning_rate=0.04,subsample=0.8,colsample_bytree=0.9,n_jobs=2); g.update(gp)
                P[k][v]=xgb.XGBRegressor(**g).fit(xgX(ab,av),yy,sample_weight=sw).predict(xgX(b.iloc[v],av))
    r=lambda p: round(float(np.sqrt(np.mean((p-yb)**2))),3)
    res={k:r(p) for k,p in P.items()}
    res['avg lgb_w10+xgb_w10']=r((P['lgb_w10']+P['xgb_w10'])/2)
    res['avg all']=r(np.mean([P[k] for k in P],axis=0))
    best=sorted(P,key=lambda k:r(P[k]))[:3]; res['avg best3']=r(np.mean([P[k] for k in best],axis=0))
    print(tgt,av,json.dumps(res),flush=True)
