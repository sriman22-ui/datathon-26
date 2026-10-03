import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(A),-1)
def ext(df,opts):
    X=[F.design(df)]; T=F._oh(df.building_type,F.TYP); B=F._oh(df.building_id,F.BLD); H=F._oh(df.hour,range(24)); we=(df.dow.values>=5).astype(float)[:,None]
    t=df.temperature.values; o=df.occupancy.values/100; p=df.previous_usage.values/50
    if 'tspl' in opts:
        S=np.column_stack([np.maximum(t-k,0) for k in [25,27,31,33]]); X.append(x(S,T)*0.5)
    if 'osp' in opts:
        S=np.column_stack([np.maximum(o-k,0) for k in [0.2,0.5,1.0,1.5,2.0]]); X.append(x(S,T)*0.5)
    if 'psp' in opts:
        S=np.column_stack([np.maximum(p-k,0) for k in [0.4,0.8,1.2,1.6]]); X.append(x(S,T)*0.5)
    if 'pH' in opts: X.append(x(p[:,None]*np.c_[df.hour.isin([22,23,0,1,2,3,4]).values,df.hour.isin([7,8,9]).values].astype(float),T))
    if 'oT_we' in opts: X.append(x(o[:,None]*we,T))
    if 'tO' in opts: X.append(x((t[:,None]-28)*o[:,None],T)*0.5)
    if 'tP' in opts: X.append(x((t[:,None]-28)*p[:,None],T)*0.5)
    if 'oP' in opts: X.append(x(o[:,None]*p[:,None],T)*0.5)
    return np.hstack(X)
def run(opts,alpha=1.0):
    P=np.zeros(len(d))
    for a,b in KFold(5,shuffle=True,random_state=0).split(d):
        P[b]=Ridge(alpha).fit(ext(d.iloc[a],opts),y[a]).predict(ext(d.iloc[b],opts))
    return np.sqrt(np.mean((P-y)**2)), np.sqrt(np.sum(w*(P-y)**2)/w.sum())
for o in [(),('tspl',),('osp',),('psp',),('pH',),('oT_we',),('tO',),('tP',),('oP',)]:
    r=run(o); print(str(o).ljust(14),round(r[0],4),round(r[1],4),flush=True)
