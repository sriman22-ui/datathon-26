import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); te=F.clean(pd.read_csv('test.csv'))
av=('t','h','o')
a=tr.dropna(subset=list(F.NUMS.values())); b=te.dropna(subset=list(F.NUMS.values())).copy()
m=Ridge(1).fit(F.design(a,av),a.previous_usage)
# in-train OOF residual for reference
oo=np.zeros(len(a))
for x,v in KFold(5,shuffle=True,random_state=0).split(a): oo[v]=a.previous_usage.values[v]-Ridge(1).fit(F.design(a.iloc[x],av),a.previous_usage.values[x]).predict(F.design(a.iloc[v],av))
b['res']=b.previous_usage-m.predict(F.design(b,av)); a=a.assign(res=oo)
print('train OOF rmse',np.sqrt((oo**2).mean()),'test rmse',np.sqrt((b.res**2).mean()),'test bias',b.res.mean())
def seg(df,name):
    df=df.assign(tb=pd.cut(df.temperature,[0,25,27,29,31,33,40]),ob=pd.cut(df.occupancy,[-1,0,50,100,150,200,250,400]),hb=pd.cut(df.hour,[-1,5,8,11,16,19,21,23]))
    for c in ['tb','ob','hb','building_type']:
        g=df.groupby(c,observed=True).res.agg(['count','mean',lambda s:np.sqrt((s**2).mean())]); g.columns=['n','bias','rmse']; print(name,c); print(g.round(2).T.to_string())
seg(a,'TRAIN'); seg(b,'TEST')
q=np.quantile(oo,[0.005,0.01,0.99,0.995]); print('train resid quantiles',q.round(1))
for lo,hi in [(q[0],q[3]),(q[1],q[2])]:
    print('test frac outside',((b.res<lo)|(b.res>hi)).mean().round(3),'n',((b.res<lo)|(b.res>hi)).sum(), 'train frac',((oo<lo)|(oo>hi)).mean().round(3))
out=b[(b.res>q[3])]; print(out.groupby('hour').size().to_dict()); print(out.building_id.value_counts().to_dict())
print(out[['building_id','hour','occupancy','previous_usage','res']].head(10))
A=a.copy(); A['S']=Ridge(1).fit(F.design(a,av),a.energy_usage).predict(F.design(a,av))  # in-sample, fine for diagnostics
ext=A[A.res>q[2]]
print('train extreme rows',len(ext),ext.groupby('hour').size().to_dict())
from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(ext[['res']],ext.energy_usage-ext.S); print('train: (y-S) vs prev-resid slope',lr.coef_,lr.intercept_)
nr=A[(A.res.abs()<15)]; lr=LinearRegression().fit(nr[['res']],nr.energy_usage-nr.S); print('normal slope',lr.coef_)
print(ext[['building_id','hour','occupancy','previous_usage','res','energy_usage','S']].head(12))
