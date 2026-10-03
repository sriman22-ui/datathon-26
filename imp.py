from ridgelib import *
import lightgbm as lgb
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
CMAP={'t':'temperature','h':'humidity','o':'occupancy','p':'previous_usage'}
def gbX(df,av):
    X=df[['hour','dow','month']+[CMAP[a] for a in av]].copy(); X['b']=pd.Categorical(df.building_id,BLD); return X
def imputer_gbm(dtr,target,av):
    m=lgb.LGBMRegressor(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1).fit(gbX(dtr,av),dtr[CMAP[target]])
    return lambda df: m.predict(gbX(df,av))
def imputer_ridge(dtr,target,av):
    def des(df):
        B=oh(df.building_id,BLD);H=oh(df.hour,range(24));W=oh(df.dow,range(7));M=oh(df.month,range(1,13))
        x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
        we=(df.dow.values>=5).astype(float)[:,None]
        parts=[B,H,W,M,x(H,B),x(we,B),x(H*we,B)]
        if av: num=np.column_stack([df[CMAP[a]].values for a in av]); parts+=[num,x(num,B)]
        return np.hstack(parts)
    m=Ridge(1.0).fit(des(dtr),dtr[CMAP[target]])
    return lambda df: m.predict(des(df))
def with_imp(miss,kind):
    full=('t','h','o','p'); av=tuple(c for c in full if c not in miss)
    def f(dtr,ytr,dte):
        dte=dte.copy()
        for mv in miss:
            imp=(imputer_gbm if kind=='gbm' else imputer_ridge)(dtr,mv,av); dte[CMAP[mv]]=imp(dte)
        return ridge_fp()(dtr,ytr,dte)
    return f
for miss in [('o',),('p',),('t',),('o','p'),('t','p')]:
    av=tuple(c for c in ('t','h','o','p') if c not in miss)
    print(miss,'direct',end=': '); evaluate(ridge_fp(avail=av),d,y)
    print(miss,'imp gbm',end=': '); evaluate(with_imp(miss,'gbm'),d,y)
    print(miss,'imp ridge',end=': '); evaluate(with_imp(miss,'ridge'),d,y)
