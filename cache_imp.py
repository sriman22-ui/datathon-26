# Precompute per-fold imputations of test-like masked validation rows (imputers fit on fold-train only)
import sys; sys.path.insert(0,'ml_pipeline')
from model_lib import *
import pickle
tr=pd.read_csv('ml_pipeline/split_train.csv'); va=pd.read_csv('ml_pipeline/split_val.csv')
d=pd.concat([tr,va]).reset_index(drop=True).dropna().reset_index(drop=True)
te=F.clean(pd.read_csv('test.csv'))
rng=np.random.default_rng(0); tm=te[list(F.NUMS.values())].isna().values; masks=tm[rng.integers(0,len(tm),len(d))]
n=len(d); idx=np.arange(n); splits=[('kf',list(KFold(5,shuffle=True,random_state=0).split(d)))]
for c,lo,hi in [('temperature',.06,.94),('occupancy',0,.90),('previous_usage',.06,.94)]:
    ql,qh=d[c].quantile([lo,hi]); t=((d[c]<ql)|(d[c]>qh)).values if lo>0 else (d[c]>qh).values
    splits.append(('ext_'+c[:4],[(idx[~t],idx[t])]))
GPf=dict(n_estimators=400,learning_rate=0.04,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
out={'d':d,'masks':masks,'splits':splits,'imp':{}}
for name,fl in splits:
    for fi,(a,b) in enumerate(fl):
        dtr=d.iloc[a]; dm=d.iloc[b].copy()
        for j,c in enumerate(F.NUMS.values()): dm.loc[dm.index[masks[b,j]],c]=np.nan
        imp=dm.copy(); keys=[tuple(r) for r in dm[list(F.NUMS.values())].isna().values]
        for pat in set(keys):
            ms=tuple(x for x,m in zip(F.FULL,pat) if m)
            if not ms: continue
            av=tuple(x for x in F.FULL if x not in ms); rows=[i for i,k in enumerate(keys) if k==pat]
            for mv in ms:
                g=lgb.LGBMRegressor(**GPf).fit(F.gbm_frame(dtr,av),dtr[F.NUMS[mv]])
                imp.iloc[rows,imp.columns.get_loc(F.NUMS[mv])]=g.predict(F.gbm_frame(dm.iloc[rows],av))
        out['imp'][(name,fi)]=(dm,imp)
    print(name,flush=True)
pickle.dump(out,open('ml_pipeline/imp_cache.pkl','wb'))
