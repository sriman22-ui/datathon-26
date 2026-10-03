import sys; sys.path.insert(0,'ml_pipeline')
import guard, artifact as A, features as F, numpy as np, pandas as pd, joblib, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
tr=pd.read_csv('ml_pipeline/split_train.csv'); va=pd.read_csv('ml_pipeline/split_val.csv'); dev=pd.concat([tr,va]).reset_index(drop=True); dev['id']='x'
te=pd.read_csv('test.csv'); y=dev.energy_usage.values
# step 13: OOF on dev with final recipe
oof=np.zeros(len(dev))
for a,b in KFold(5,shuffle=True,random_state=0).split(dev):
    art=A.build(dev.iloc[a],unlabeled=te,c_mode='blend'); oof[b]=A.predict(art,dev.iloc[b])
d=F.clean(dev).assign(pred=oof,res=y-oof)
rm=lambda s: np.sqrt((s**2).mean())
print('dev OOF RMSE',rm(d.res),'MAE',mean_absolute_error(y,oof),'R2',r2_score(y,oof))
f,ax=plt.subplots(2,2,figsize=(14,8))
g=d.groupby('building_id').res.apply(rm); ax[0,0].bar(g.index,g.values); ax[0,0].set_title('OOF RMSE by building'); plt.setp(ax[0,0].get_xticklabels(),rotation=45)
g=d.groupby('hour').res.apply(rm); ax[0,1].bar(g.index,g.values); ax[0,1].set_title('OOF RMSE by hour')
lv=pd.qcut(d.pred,10); g=d.groupby(lv,observed=True).res.agg([rm,'mean']); ax[1,0].plot(range(10),g.iloc[:,0],'o-',label='RMSE'); ax[1,0].plot(range(10),g.iloc[:,1],'s-',label='bias'); ax[1,0].legend(); ax[1,0].set_xlabel('prediction decile'); ax[1,0].set_title('Error grows with usage level (noise ~ proportional)')
segs={'complete':d[list(F.NUMS.values())].notna().all(1),'any missing':d[list(F.NUMS.values())].isna().any(axis=1),'hot >33.5C':d.temperature>33.5,'humid >93%':d.humidity>93,'prev<14 (night)':d.previous_usage<14,'occ>250':d.occupancy>250,'hours 22-23':d.hour.isin([22,23])}
vals=[rm(d.res[m]) for m in segs.values()]; ax[1,1].barh(list(segs.keys()),vals); ax[1,1].set_title('OOF RMSE by segment')
for k,m in segs.items(): print(k,m.sum(),round(rm(d.res[m]),3),'bias',round(d.res[m].mean(),3))
guard.fig(13,'error_analysis',f,f"Errors are flat across hours and buildings except higher for high-usage science labs and late-evening transition hours; RMSE rises with predicted level (multiplicative noise), bias stays ~0 in every decile; hot/edge segments are slightly worse only because their usage level is higher.")
d.sort_values('res').head(8).to_csv('ml_pipeline/worst_under.csv'); d.sort_values('res').tail(8).to_csv('ml_pipeline/worst_over.csv')
# step 14: final test, touched once
ts=pd.read_csv('ml_pipeline/split_test.csv')
art=A.build(dev,unlabeled=te,c_mode='blend')
def rmse(a,b): return float(np.sqrt(np.mean((a-b)**2)))
score=guard.final_test(lambda X: A.predict(art,X.assign(id='x')),ts,target='energy_usage',metric_fn=rmse)
pt=A.predict(art,ts.drop(columns=['energy_usage']).assign(id='x'))
print('FINAL internal test RMSE',score,'MAE',mean_absolute_error(ts.energy_usage,pt),'R2',r2_score(ts.energy_usage,pt))
