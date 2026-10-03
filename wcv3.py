import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge, HuberRegressor
from scipy.stats import kurtosis
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(A),-1)
def desBH(df):
    B=F._oh(df.building_id,F.BLD);H=F._oh(df.hour,range(24)); return np.hstack([F.design(df),x(H,B)*0.5])
fold=list(KFold(5,shuffle=True,random_state=0).split(d))
def cv(fn):
    P=np.zeros(len(d))
    for a,b in fold: P[b]=fn(d.iloc[a],y[a],d.iloc[b])
    return P
wr=lambda p: np.sqrt(np.sum(w*(p-y)**2)/w.sum()); r=lambda p: np.sqrt(np.mean((p-y)**2))
P={}
P['base']=cv(lambda A,ya,B: Ridge(1).fit(F.design(A),ya).predict(F.design(B)))
print('resid kurtosis',kurtosis(y-P['base']))
P['BH']=cv(lambda A,ya,B: Ridge(1).fit(desBH(A),ya).predict(desBH(B)))
def wls(A,ya,B):
    m0=Ridge(1).fit(F.design(A),ya); mu=np.clip(m0.predict(F.design(A)),10,None)
    return Ridge(1).fit(F.design(A),ya,sample_weight=(50/mu)**2).predict(F.design(B))
P['wls2']=cv(wls)
def wls1(A,ya,B):
    m0=Ridge(1).fit(F.design(A),ya); mu=np.clip(m0.predict(F.design(A)),10,None)
    return Ridge(1).fit(F.design(A),ya,sample_weight=(50/mu)).predict(F.design(B))
P['wls1']=cv(wls1)
def sq(A,ya,B):
    m=Ridge(1).fit(F.design(A),np.sqrt(ya)); s2=np.var(np.sqrt(ya)-m.predict(F.design(A))); p=m.predict(F.design(B)); return p**2+s2
P['sqrt']=cv(sq)
def hub(A,ya,B):
    XA=F.design(A); m=HuberRegressor(epsilon=1.6,alpha=1e-3,max_iter=500).fit(XA,ya); return m.predict(F.design(B))
P['huber']=cv(hub)
for k,p in P.items(): print(k.ljust(8),round(r(p),4),round(wr(p),4))
for combo in [('base','BH'),('base','wls1'),('base','sqrt'),('base','BH','wls1','sqrt')]:
    p=np.mean([P[c] for c in combo],axis=0); print('avg',combo,round(r(p),4),round(wr(p),4))
