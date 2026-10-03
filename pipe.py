from ridgelib import *
import lightgbm as lgb
CMAP={'t':'temperature','h':'humidity','o':'occupancy','p':'previous_usage'}
FULL=('t','h','o','p')
def gbX(df,av):
    X=df[['hour','dow','month']+[CMAP[a] for a in av]].copy(); X['b']=pd.Categorical(df.building_id,BLD); return X
GP=dict(n_estimators=600,learning_rate=0.03,num_leaves=31,min_child_samples=20,subsample=0.8,subsample_freq=1,verbose=-1)
class Model:
    def __init__(self,opt=None,alpha=1.0,mode='imp'):
        self.opt=opt;self.alpha=alpha;self.mode=mode
    def fit(self,d,y):
        self.d=d;self.y=y
        self.main=Ridge(self.alpha).fit(design(d,self.opt),y)
        self.cache={}
        return self
    def _imp(self,target,av):
        k=(target,av)
        if k not in self.cache:
            self.cache[k]=lgb.LGBMRegressor(**GP).fit(gbX(self.d,av),self.d[CMAP[target]])
        return self.cache[k]
    def _direct(self,av):
        k=('direct',av)
        if k not in self.cache: self.cache[k]=Ridge(self.alpha).fit(design(self.d,self.opt,av),self.y)
        return self.cache[k]
    def predict(self,df):
        df=df.reset_index(drop=True); out=np.zeros(len(df))
        miss=df[[CMAP[a] for a in FULL]].isna().values
        pats={}
        for i,row in enumerate(miss): pats.setdefault(tuple(row),[]).append(i)
        for pat,idx in pats.items():
            sub=df.iloc[idx].copy()
            ms=tuple(a for a,m in zip(FULL,pat) if m); av=tuple(a for a in FULL if a not in ms)
            if not ms: out[idx]=self.main.predict(design(sub,self.opt)); continue
            if self.mode in('imp','blend'):
                s2=sub.copy()
                for mv in ms: s2[CMAP[mv]]=self._imp(mv,av).predict(gbX(sub,av))
                pi=self.main.predict(design(s2,self.opt))
            if self.mode in('direct','blend'):
                pd_=self._direct(av).predict(design(sub,self.opt,av))
            out[idx]= pi if self.mode=='imp' else pd_ if self.mode=='direct' else 0.5*(pi+pd_)
        return out
