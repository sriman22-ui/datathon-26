import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
base=dict(F.SCALE_V8)
def mk(ch):
    def des(df):
        F.SCALE_V8=dict(base,**ch); return F.design_v8(df)
    return des
D={'v6':lambda df:F.design(df),'v8':mk({}),'v8TH.5':mk({'TH':0.5})}
res={}
for nm,des in D.items():
    P=np.zeros(len(d))
    for s in range(3):
        for a,b in KFold(5,shuffle=True,random_state=10+s).split(d): P[b]+=Ridge(1).fit(des(d.iloc[a]),y[a]).predict(des(d.iloc[b]))/3
    E=[]
    for c,lo,hi in [('temperature',.06,.94),('occupancy',0,.9),('previous_usage',.06,.94)]:
        ql,qh=d[c].quantile([lo,hi]); t=((d[c]<ql)|(d[c]>qh)).values if lo>0 else (d[c]>qh).values
        E.append((t,Ridge(1).fit(des(d[~t]),y[~t]).predict(des(d[t]))))
    res[nm]=(P,E)
def rep(nm,P,E):
    print(nm.ljust(14),round(np.sqrt(np.mean((P-y)**2)),4),round(np.sqrt(np.sum(w*(P-y)**2)/w.sum()),4),[round(np.sqrt(np.mean((p-y[t])**2)),4) for t,p in E])
for k,(P,E) in res.items(): rep(k,P,E)
for combo in [('v6','v8'),('v6','v8TH.5'),('v8','v8TH.5'),('v6','v8','v8TH.5')]:
    P=np.mean([res[c][0] for c in combo],axis=0); E=[(res[combo[0]][1][i][0],np.mean([res[c][1][i][1] for c in combo],axis=0)) for i in range(3)]
    rep('+'.join(combo),P,E)
