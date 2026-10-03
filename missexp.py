import sys; sys.path.insert(0,'ml_pipeline')
from fast_eval import *
tw=pd.read_csv('train_advw.csv')
k=['building_id','hour','dow','month','temperature','humidity','occupancy','previous_usage','energy_usage']
w=d.merge(tw[k+['w']],on=k,how='left').w.values
M=C['masks']; fl=C['splits'][0][1]
GPd=dict(n_estimators=500,learning_rate=0.03,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=2)
out={k_:np.full(len(d),np.nan) for k_ in ['base','imp_w','direct_lgb','direct_lgb_w','resid_lgb']}
for fi,(a,b) in enumerate(fl):
    dtr=d.iloc[a]; ytr=y[a]; wtr=w[a]
    main=Ridge(1).fit(F.design(dtr),ytr)
    dm,imp=C['imp'][('kf',fi)]
    out['base'][b]=main.predict(F.design(imp))
    keys=[tuple(r) for r in dm[list(F.NUMS.values())].isna().values]
    # OOF ridge residual on fold-train for residual model
    for pat in set(keys):
        if not any(pat): 
            rows=[i for i,kk in enumerate(keys) if kk==pat]
            for kname in ['imp_w','direct_lgb','direct_lgb_w','resid_lgb']: out[kname][b[rows]]=out['base'][b[rows]]
            continue
        ms=tuple(x for x,mm in zip(F.FULL,pat) if mm); av=tuple(x for x in F.FULL if x not in ms)
        rows=np.array([i for i,kk in enumerate(keys) if kk==pat]); sub=dm.iloc[rows]
        s2=sub.copy()
        for mv in ms:
            g=lgb.LGBMRegressor(**{**GPd,'num_leaves':31,'n_estimators':400,'learning_rate':0.04}).fit(F.gbm_frame(dtr,av),dtr[F.NUMS[mv]],sample_weight=wtr)
            s2[F.NUMS[mv]]=g.predict(F.gbm_frame(sub,av))
        out['imp_w'][b[rows]]=main.predict(F.design(s2))
        gd=lgb.LGBMRegressor(**GPd).fit(F.gbm_frame(dtr,av),ytr); out['direct_lgb'][b[rows]]=gd.predict(F.gbm_frame(sub,av))
        gdw=lgb.LGBMRegressor(**GPd).fit(F.gbm_frame(dtr,av),ytr,sample_weight=wtr); out['direct_lgb_w'][b[rows]]=gdw.predict(F.gbm_frame(sub,av))
        # residual correction: direct ridge on available + lgb on its residual
        dr=Ridge(1).fit(F.design(dtr,av),ytr); res=ytr-dr.predict(F.design(dtr,av))
        gr=lgb.LGBMRegressor(**{**GPd,'n_estimators':200}).fit(F.gbm_frame(dtr,av),res)
        out['resid_lgb'][b[rows]]=dr.predict(F.design(sub,av))+gr.predict(F.gbm_frame(sub,av))
    print(fi,flush=True)
m=M.any(1)
for k_,p in out.items():
    print(k_.ljust(13),'missing rows: rmse',round(rmse(p[m],y[m]),3),'w-rmse',round(np.sqrt(np.sum(w[m]*(p[m]-y[m])**2)/w[m].sum()),3))
for a_ in [0.3,0.5]:
    for k_ in ['direct_lgb','resid_lgb','imp_w']:
        p=(1-a_)*out['base']+a_*out[k_]; print('blend base+',k_,a_,round(rmse(p[m],y[m]),3),round(np.sqrt(np.sum(w[m]*(p[m]-y[m])**2)/w[m].sum()),3))
np.save('missexp.npy',np.vstack([out[k_] for k_ in out]))
