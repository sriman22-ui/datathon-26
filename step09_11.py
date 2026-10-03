import sys; sys.path.insert(0,'ml_pipeline')
from model_lib import *
import json, time
from sklearn.linear_model import LinearRegression
tr=pd.read_csv('ml_pipeline/split_train.csv'); va=pd.read_csv('ml_pipeline/split_val.csv')
dev=pd.concat([tr,va]).reset_index(drop=True); d=dev.dropna().reset_index(drop=True)
te=F.clean(pd.read_csv('test.csv'))
rng=np.random.default_rng(0); tm=te[list(F.NUMS.values())].isna().values; masks=tm[rng.integers(0,len(tm),len(d))]
log=open('ml_pipeline/runs.jsonl','a')
def run(name,make,**kw):
    t=time.time(); r,_=harness(make,d,masks,**kw); r['name']=name; r['sec']=round(time.time()-t)
    print(json.dumps(r)); log.write(json.dumps(r)+'\n'); log.flush(); return r
# baselines
class MeanModel:
    def fit(self,df,y): self.m=np.mean(y); return self
    def predict(self,df): return np.full(len(df),self.m)
class PlainLinear:
    def fit(self,df,y):
        X=self._X(df); self.med=X.median(); self.m=LinearRegression().fit(X.fillna(self.med),y); return self
    def _X(self,df):
        X=pd.get_dummies(df[['building_id','day_of_week']].astype(str)); X=X.reindex(columns=[f'building_id_{b}' for b in F.BLD]+[f'day_of_week_{d}' for d in F.DAYS],fill_value=0).astype(float)
        for c in ['hour','month','temperature','humidity','occupancy','previous_usage']: X[c]=df[c].values
        return X
    def predict(self,df): return self.m.predict(self._X(df).fillna(self.med))
class LGBPlain:
    def fit(self,df,y): self.m=lgb.LGBMRegressor(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=4).fit(F.gbm_frame(df,F.FULL),y); return self
    def predict(self,df): return self.m.predict(F.gbm_frame(df,F.FULL))
which=sys.argv[1]
if which=='base':
    pass
    run('ridge_struct_gbmimp',lambda: EnergyModel())
if which=='tune':
    for a in [0.3,3.0]: run(f'ridge a={a}',lambda: EnergyModel(alpha=a))
    for h in [29.0,31.0,None]: run(f'ridge hinge={h}',lambda: EnergyModel(hinge=h))
    run('p_mode=direct',lambda: EnergyModel(p_mode='direct'))
    run('p_mode=blend',lambda: EnergyModel(p_mode='blend'))
    run('wls1',lambda: EnergyModel(wls=1))
