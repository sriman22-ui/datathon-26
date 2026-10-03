import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
idx=np.arange(len(d))
for nm,des in [('v6 design',lambda df: F.design(df)),('v8 design',F.design_v8)]:
    out=[]
    P=np.zeros(len(d))
    for s in range(3):
        for a,b in KFold(5,shuffle=True,random_state=10+s).split(d): P[b]+=Ridge(1).fit(des(d.iloc[a]),y[a]).predict(des(d.iloc[b]))/3
    out.append(('kf(new seeds)',np.sqrt(np.mean((P-y)**2)),np.sqrt(np.sum(w*(P-y)**2)/w.sum())))
    for c,lo,hi in [('temperature',.06,.94),('occupancy',0,.9),('previous_usage',.06,.94),('humidity',.05,.95)]:
        ql,qh=d[c].quantile([lo,hi]); t=((d[c]<ql)|(d[c]>qh)).values if lo>0 else (d[c]>qh).values
        p=Ridge(1).fit(des(d[~t]),y[~t]).predict(des(d[t])); out.append(('ext_'+c[:4],np.sqrt(np.mean((p-y[t])**2))))
    print(nm,[(o[0],round(o[1],4))+((round(o[2],4),) if len(o)>2 else ()) for o in out])
