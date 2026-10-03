import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
base=dict(F.SCALE_V8)
variants={'v8':{},'no o_hb_T':{'o_hb_T':0},'num1':{'num':1.0},'no o_hb_T,num1':{'o_hb_T':0,'num':1.0},'no Bwe':{'Bwe':0},'no t_hb':{'t_hb':0}}
for nm,ch in variants.items():
    F.SCALE_V8=dict(base,**ch); des=F.design_v8
    P=np.zeros(len(d))
    for s in range(3):
        for a,b in KFold(5,shuffle=True,random_state=10+s).split(d): P[b]+=Ridge(1).fit(des(d.iloc[a]),y[a]).predict(des(d.iloc[b]))/3
    ext=[]
    for c,lo,hi in [('temperature',.06,.94),('occupancy',0,.9),('previous_usage',.06,.94)]:
        ql,qh=d[c].quantile([lo,hi]); t=((d[c]<ql)|(d[c]>qh)).values if lo>0 else (d[c]>qh).values
        p=Ridge(1).fit(des(d[~t]),y[~t]).predict(des(d[t])); ext.append(round(np.sqrt(np.mean((p-y[t])**2)),4))
    print(nm.ljust(16),round(np.sqrt(np.mean((P-y)**2)),4),round(np.sqrt(np.sum(w*(P-y)**2)/w.sum()),4),ext,flush=True)
