import sys; sys.path.insert(0,'ml_pipeline')
import numpy as np, pandas as pd, lightgbm as lgb
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
import features as F
GP=dict(n_estimators=300,learning_rate=0.05,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1,n_jobs=4)
class EnergyModel:
    def __init__(self,alpha=1.0,hinge=30.0,imp='gbm',p_mode='imp',wls=False,gp=None):
        self.alpha=alpha;self.hinge=hinge;self.imp=imp;self.p_mode=p_mode;self.wls=wls;self.gp={**GP,**(gp or {})}
    def fit(self,df,y):
        ok=df[list(F.NUMS.values())].notna().all(1).values
        self.d=df[ok].reset_index(drop=True); self.y=np.asarray(y)[ok]
        w=None
        if self.wls:
            m0=Ridge(self.alpha).fit(F.design(self.d,hinge=self.hinge),self.y); mu=np.clip(m0.predict(F.design(self.d,hinge=self.hinge)),10,None)
            w=(40.0/mu)**self.wls
        self.main=Ridge(self.alpha).fit(F.design(self.d,hinge=self.hinge),self.y,sample_weight=w)
        self.w=w; self.imps={}; self.directs={}
        return self
    def _imputer(self,target,av):
        k=(target,av)
        if k not in self.imps:
            if self.imp in('gbm','mix'):
                g=lgb.LGBMRegressor(**self.gp).fit(F.gbm_frame(self.d,av),self.d[F.NUMS[target]])
            else: g=None
            self.imps[k]=g
        return self.imps[k]
    def _direct(self,av):
        if av not in self.directs: self.directs[av]=Ridge(self.alpha).fit(F.design(self.d,av,self.hinge),self.y,sample_weight=self.w)
        return self.directs[av]
    def predict(self,df):
        df=df.reset_index(drop=True); out=np.zeros(len(df))
        miss=df[list(F.NUMS.values())].isna().values
        keys=[tuple(r) for r in miss]
        for pat in set(keys):
            idx=[i for i,k in enumerate(keys) if k==pat]; sub=df.iloc[idx].copy()
            ms=tuple(a for a,m in zip(F.FULL,pat) if m); av=tuple(a for a in F.FULL if a not in ms)
            if not ms: out[idx]=self.main.predict(F.design(sub,hinge=self.hinge)); continue
            s2=sub.copy()
            for mv in ms: s2[F.NUMS[mv]]=self._imputer(mv,av).predict(F.gbm_frame(sub,av))
            pi=self.main.predict(F.design(s2,hinge=self.hinge))
            if self.p_mode=='imp' or 'p' not in ms: out[idx]=pi
            else:
                pdr=self._direct(av).predict(F.design(sub,av,self.hinge))
                out[idx]=pdr if self.p_mode=='direct' else 0.5*(pi+pdr)
        return out

def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def harness(make,d,masks,ext=True,seed=0):
    """d: dev frame (complete rows). masks: boolean (n,4) test-like missingness to apply on validation rows."""
    y=d.energy_usage.values; n=len(d); idx=np.arange(n); res={}
    splits=[('kf',list(KFold(5,shuffle=True,random_state=seed).split(d)))]
    if ext:
        for c,lo,hi in [('temperature',.06,.94),('occupancy',0,.90),('previous_usage',.06,.94)]:
            ql,qh=d[c].quantile([lo,hi]); t=((d[c]<ql)|(d[c]>qh)).values if lo>0 else (d[c]>qh).values
            splits.append(('ext_'+c[:4],[(idx[~t],idx[t])]))
    oof=None
    for name,fl in splits:
        P=np.full(n,np.nan); PM=np.full(n,np.nan)
        for a,b in fl:
            m=make().fit(d.iloc[a],y[a]); P[b]=m.predict(d.iloc[b])
            dm=d.iloc[b].copy()
            for j,c in enumerate(F.NUMS.values()): dm.loc[dm.index[masks[b,j]],c]=np.nan
            PM[b]=m.predict(dm)
        k=~np.isnan(P); res[name]=round(rmse(P[k],y[k]),4); res[name+'_miss']=round(rmse(PM[k],y[k]),4)
        if name=='kf': oof=(P,PM)
    return res,oof
