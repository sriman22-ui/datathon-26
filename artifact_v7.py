"""Model artifact v5 (v7 candidate): WLS structural ridge + transductive imputer ensembles (LightGBM x3 + XGBoost,
train + unlabeled test features, test weight 10) + uncertainty-scaled missingness offsets + bagged NaN-native
LightGBM blended in for rows missing BOTH occupancy and previous_usage."""
import sys; sys.path.insert(0,'ml_pipeline')
import itertools, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import features as F
BASE=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2,random_state=0)
LGB_CFGS=[BASE, dict(BASE,min_child_samples=40,random_state=1), dict(BASE,num_leaves=15,n_estimators=700,learning_rate=0.03,random_state=2)]
XGB_CFG=dict(n_estimators=500,learning_rate=0.04,max_depth=6,subsample=0.8,colsample_bytree=0.9,n_jobs=2,random_state=0)
TREE_GP=dict(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2)
OFFSETS={'t':-0.27,'h':0.18,'o':0.33,'p':0.79}      # y-units, learned OOF on train's real NaN rows (single-missing)
SIGMA_TRAIN={'t':0.83,'h':4.2,'o':22.2,'p':7.5}     # train CV imputation sd of each feature given all others
def xgb_frame(df,av):
    X=F.gbm_frame(df,av).copy(); X['b']=X.b.cat.codes; return X
def impute(models,df,av):
    ps=[m.predict(F.gbm_frame(df,av)) for m in models[:-1]]+[models[-1].predict(xgb_frame(df,av))]
    return np.mean(ps,axis=0)
def _fit_imp(sub,mv,av):
    y=sub[F.NUMS[mv]].values; w=sub._w.values
    ms=[lgb.LGBMRegressor(**c).fit(F.gbm_frame(sub,av),y,sample_weight=w) for c in LGB_CFGS]
    ms.append(xgb.XGBRegressor(**XGB_CFG).fit(xgb_frame(sub,av),y,sample_weight=w))
    return ms
def _test_sigma(u,mv,av):
    """sd of imputation error measured on the unlabeled test rows where mv is observed (3-fold, quick LightGBM)."""
    sub=u.dropna(subset=[F.NUMS[mv]]+[F.NUMS[a] for a in av]).reset_index(drop=True); y=sub[F.NUMS[mv]].values; r=np.zeros(len(sub))
    for a,b in KFold(3,shuffle=True,random_state=0).split(sub):
        r[b]=y[b]-lgb.LGBMRegressor(**dict(BASE,n_estimators=250)).fit(F.gbm_frame(sub.iloc[a],av),y[a]).predict(F.gbm_frame(sub.iloc[b],av))
    return float(np.sqrt(np.mean(r**2)))
def build(df,unlabeled=None,test_weight=10.0,alpha=1.0,hinge=29.0,c_mode='blend',n_bag=5,wls=True):
    df=F.clean(df); d=df.dropna(subset=list(F.NUMS.values())).reset_index(drop=True)
    X=F.design(d,F.FULL,hinge); y=d.energy_usage.values; w=None
    if wls:   # noise grows with usage level -> weight 1/mu (variance ~ mu)
        mu=np.clip(Ridge(alpha).fit(X,y).predict(X),10,None); w=50.0/mu
    main=Ridge(alpha).fit(X,y,sample_weight=w)
    feat=df.drop(columns=['energy_usage']).assign(_w=1.0)
    u=None
    if unlabeled is not None:
        u=F.clean(unlabeled).assign(_w=test_weight); feat=pd.concat([feat,u[feat.columns]],ignore_index=True)
    imps={}; offs={}
    for k in range(1,5):
        for ms in itertools.combinations(F.FULL,k):
            av=tuple(a for a in F.FULL if a not in ms); off=0.0
            for mv in ms:
                sub=feat.dropna(subset=[F.NUMS[mv]]+[F.NUMS[a] for a in av])
                imps[(mv,av)]=_fit_imp(sub,mv,av)
                ratio=_test_sigma(u,mv,av)/SIGMA_TRAIN[mv] if u is not None else 1.0
                off+=OFFSETS[mv]*ratio
            offs[ms]=off
    trees=[lgb.LGBMRegressor(**dict(TREE_GP,random_state=s,bagging_seed=s,feature_fraction_seed=s)).fit(F.gbm_frame(df,F.FULL),df.energy_usage.values) for s in range(n_bag)]
    return {'main':main,'imputers':imps,'trees':trees,'hinge':hinge,'offsets':offs,'c_mode':c_mode,'version':'ridge-struct-v5'}
def predict(art,df,c_mode=None):
    c_mode=c_mode or art['c_mode']
    df=F.clean(df).reset_index(drop=True); out=np.zeros(len(df))
    miss=df[list(F.NUMS.values())].isna().values; keys=[tuple(r) for r in miss]
    for pat in set(keys):
        rows=np.array([i for i,k in enumerate(keys) if k==pat]); sub=df.iloc[rows].copy()
        ms=tuple(a for a,m in zip(F.FULL,pat) if m); av=tuple(a for a in F.FULL if a not in ms)
        for mv in ms: sub[F.NUMS[mv]]=impute(art['imputers'][(mv,av)],df.iloc[rows],av)
        p=art['main'].predict(F.design(sub,F.FULL,art['hinge']))
        if ms: p=p+art['offsets'][ms]
        if 'o' in ms and 'p' in ms and c_mode!='ridge':
            t=np.mean([m.predict(F.gbm_frame(df.iloc[rows],F.FULL)) for m in art['trees']],axis=0)
            p= t if c_mode=='tree' else 0.5*(p+t)
        out[rows]=p
    return out
