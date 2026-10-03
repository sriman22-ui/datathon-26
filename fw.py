import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.model_selection import KFold
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
BLD=['ADM_A','BUS_A','BUS_B','ENG_A','ENG_B','LEC_A','LIB_A','RES_A','RES_B','SCI_A','SCI_B','SPT_A']
NUM=['temperature','humidity','occupancy','previous_usage']
def load():
    tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
    for df in (tr,te): df['dow']=df.day_of_week.map({d:i for i,d in enumerate(D)})
    return tr,te
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
def folds(d,seed=0,k=5):
    out=[('kf',list(KFold(k,shuffle=True,random_state=seed).split(d)))]
    n=len(d); idx=np.arange(n)
    for c,lo,hi in [('temperature',.06,.94),('occupancy',0,.90),('previous_usage',.06,.94),('humidity',.05,.95)]:
        ql,qh=d[c].quantile([lo,hi]); te=((d[c]<ql)|(d[c]>qh)).values if lo>0 else (d[c]>qh).values
        out.append(('ext_'+c[:4],[(idx[~te],idx[te])]))
    return out
def evaluate(fitpred,d,y,verbose=True,seed=0):
    res={}
    for name,fl in folds(d,seed):
        errs=[];P=np.full(len(y),np.nan)
        for a,b in fl:
            P[b]=fitpred(d.iloc[a],y[a],d.iloc[b])
        m=~np.isnan(P); res[name]=rmse(P[m],y[m])
        if name=='kf': oof=P
    if verbose: print('  '.join(f'{k}={v:.4f}' for k,v in res.items()))
    return res,oof
