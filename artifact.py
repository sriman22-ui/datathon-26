"""Model artifact v3: structural ridge + transductive (train+test features) GBM imputers + missingness offsets
+ NaN-native tree model for rows missing BOTH occupancy and previous_usage."""
import sys; sys.path.insert(0,'ml_pipeline')
import itertools, numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import Ridge
import features as F
IMP_GP=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2,random_state=0)
TREE_GP=dict(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2,random_state=0)
OFFSETS={'t':-0.27,'h':0.18,'o':0.33,'p':0.79}
def xgb_frame(df,av):
    X=F.gbm_frame(df,av).copy(); X['b']=X.b.cat.codes; return X
def impute(pair,df,av):
    return 0.5*(pair[0].predict(F.gbm_frame(df,av))+pair[1].predict(xgb_frame(df,av)))

def build(df,unlabeled=None,test_weight=10.0,alpha=1.0,hinge=29.0,offsets=OFFSETS,c_mode='tree'):
    df=F.clean(df); d=df.dropna(subset=list(F.NUMS.values())).reset_index(drop=True)
    main=Ridge(alpha).fit(F.design(d,F.FULL,hinge),d.energy_usage.values)
    main8=Ridge(alpha).fit(F.design_v8(d,hinge),d.energy_usage.values)
    feat=df.drop(columns=['energy_usage']).assign(_w=1.0)
    if unlabeled is not None:
        u=F.clean(unlabeled).assign(_w=test_weight); feat=pd.concat([feat,u[feat.columns]],ignore_index=True)
    imps={}
    for k in range(1,5):
        for ms in itertools.combinations(F.FULL,k):
            av=tuple(a for a in F.FULL if a not in ms)
            for mv in ms:
                sub=feat.dropna(subset=[F.NUMS[mv]]+[F.NUMS[a] for a in av])
                g1=lgb.LGBMRegressor(**IMP_GP).fit(F.gbm_frame(sub,av),sub[F.NUMS[mv]].values,sample_weight=sub._w.values)
                g2=xgb.XGBRegressor(n_estimators=500,learning_rate=0.04,max_depth=6,subsample=0.8,colsample_bytree=0.9,n_jobs=2,random_state=0).fit(xgb_frame(sub,av),sub[F.NUMS[mv]].values,sample_weight=sub._w.values)
                imps[(mv,av)]=(g1,g2)
    tree=lgb.LGBMRegressor(**TREE_GP).fit(F.gbm_frame(df,F.FULL),df.energy_usage.values)
    return {'main':main,'main8':main8,'imputers':imps,'tree':tree,'hinge':hinge,'offsets':dict(offsets),'c_mode':c_mode,'version':'ridge-struct-v6'}
def predict(art,df,c_mode=None,use_offsets=True):
    c_mode=c_mode or art['c_mode']
    df=F.clean(df).reset_index(drop=True); out=np.zeros(len(df))
    miss=df[list(F.NUMS.values())].isna().values; keys=[tuple(r) for r in miss]
    for pat in set(keys):
        rows=np.array([i for i,k in enumerate(keys) if k==pat]); sub=df.iloc[rows].copy()
        ms=tuple(a for a,m in zip(F.FULL,pat) if m); av=tuple(a for a in F.FULL if a not in ms)
        for mv in ms: sub[F.NUMS[mv]]=impute(art['imputers'][(mv,av)],df.iloc[rows],av)
        p=0.5*(art['main'].predict(F.design(sub,F.FULL,art['hinge']))+art['main8'].predict(F.design_v8(sub,art['hinge'])))
        if use_offsets: p=p+sum(art['offsets'][m] for m in ms)
        if 'o' in ms and 'p' in ms and c_mode!='ridge':
            t=art['tree'].predict(F.gbm_frame(df.iloc[rows],F.FULL))
            p= t if c_mode=='tree' else 0.5*(p+t)
        out[rows]=p
    return out
