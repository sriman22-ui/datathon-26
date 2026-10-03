import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
av=('t','h','o')
a=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); b=te.dropna(subset=list(F.NUMS.values())).copy()
m=Ridge(1).fit(F.design(a,av),a.previous_usage)
oo=np.zeros(len(a))
for x,v in KFold(5,shuffle=True,random_state=0).split(a): oo[v]=a.previous_usage.values[v]-Ridge(1).fit(F.design(a.iloc[x],av),a.previous_usage.values[x]).predict(F.design(a.iloc[v],av))
a['res']=oo; b['res']=b.previous_usage-m.predict(F.design(b,av)); b['phat']=b.previous_usage-b.res
for name,df in [('train',a),('test',b)]:
    x=df[df.hour.between(0,11)]
    print(name,'hours0-11 res quantiles',np.round(np.quantile(x.res,[.01,.05,.1,.25,.5,.75,.9,.95,.99]),1))
x=b[b.hour.between(0,11)].sort_values('res')
print(x.head(8)[['building_id','hour','day_of_week','temperature','occupancy','previous_usage','phat','res']])
print(x.tail(8)[['building_id','hour','day_of_week','temperature','occupancy','previous_usage','phat','res']])
oof=np.zeros(len(a))
for x,v in KFold(5,shuffle=True,random_state=0).split(a): oof[v]=Ridge(1).fit(F.design(a.iloc[x]),a.energy_usage.values[x]).predict(F.design(a.iloc[v]))
a['oof']=oof; a['phat']=a.previous_usage-a.res
for cond,name in [((a.building_id.isin(['ENG_A','ENG_B']))&(a.hour<=5)&(a.res>20),'ENG night high prev'),((a.building_id=='SCI_A')&(a.hour.between(7,9))&(a.res<-20),'SCI morning low prev'),((a.res>25),'all res>25'),((a.res<-20),'all res<-20')]:
    x=a[cond]; print(name,len(x),'y-oof mean',round((x.energy_usage-x.oof).mean(),2),'rmse',round(np.sqrt(((x.energy_usage-x.oof)**2).mean()),2))
    print(x[['building_id','hour','occupancy','previous_usage','phat','energy_usage','oof']].head(6).round(1).to_string())
print('=== prev-model bias by building, stable hours 10-20 only')
for name,df in [('train(OOF)',a),('test',b)]:
    x=df[df.hour.between(10,20)]
    g=x.groupby('building_id').res.agg(['count','mean']); print(name); print(g.round(2).T.to_string())
