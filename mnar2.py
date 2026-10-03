import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); y=tr.energy_usage.values
for tgt,av in [('previous_usage',('t','h','o')),('occupancy',('t','h','p'))]:
    ok=tr[[F.NUMS[a] for a in av]].notna().all(1).values
    d=tr[ok].reset_index(drop=True); yy=d.energy_usage.values
    res=np.zeros(len(d))
    for a,b in KFold(5,shuffle=True,random_state=0).split(d):
        A=d.iloc[a]; 
        m=Ridge(1).fit(F.design(A,av),A.energy_usage); res[b]=yy[b]-m.predict(F.design(d.iloc[b],av))
    mis=d[tgt].isna().values
    print(tgt,'missing n',mis.sum(),'resid mean miss',res[mis].mean().round(3),'sd',res[mis].std().round(2),'| nonmiss mean',res[~mis].mean().round(3),'sd',res[~mis].std().round(2))
    print('   quantiles miss',np.round(np.quantile(res[mis],[.05,.25,.5,.75,.95]),2),' nonmiss',np.round(np.quantile(res[~mis],[.05,.25,.5,.75,.95]),2))
