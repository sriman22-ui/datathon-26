from ridgelib import *
from sklearn.model_selection import KFold
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
oof=np.zeros(len(y))
for a,b in KFold(5,shuffle=True,random_state=0).split(d): oof[b]=ridge_fp()(d.iloc[a],y[a],d.iloc[b])
nop=np.zeros(len(y))
for a,b in KFold(5,shuffle=True,random_state=0).split(d): nop[b]=ridge_fp(avail=('t','h','o'))(d.iloc[a],y[a],d.iloc[b])
d['res']=y-oof; d['nop']=nop; d['resnop']=y-nop
hot=d.temperature>33.5; hum=d.humidity>93; low=d.previous_usage<14
for n,m in [('hot',hot),('hum',hum),('low',low),('rest',~(hot|hum|low))]:
    print(n,m.sum(),'ridge rmse',rmse(oof[m],y[m]),'bias',d.res[m].mean().round(2),'noprev rmse',rmse(nop[m],y[m]))
d['ratio']=d.previous_usage/d.nop
print(d.ratio.describe()); print(d[low].ratio.describe())
print((d.ratio<0.6).sum(), ((d.ratio<0.6)&low).sum())
print(d.sort_values('res').head(10)[['building_id','hour','temperature','humidity','occupancy','previous_usage','energy_usage','res','ratio']])
print(d.sort_values('res').tail(10)[['building_id','hour','temperature','humidity','occupancy','previous_usage','energy_usage','res','ratio']])
