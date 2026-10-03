import sys; sys.path.insert(0,'ml_pipeline')
from fast_eval import *
res,(P,PM)=evaluate(ridge_fit(hinge=29.0),'x',log=False)
tw=pd.read_csv('train_advw.csv')
k=['building_id','hour','dow','month','temperature','humidity','occupancy','previous_usage','energy_usage']
dd=d.merge(tw[k+['w']],on=k,how='left'); w=dd.w.values; print(np.isnan(w).sum(), len(w), len(d))
wr=lambda p: np.sqrt(np.sum(w*(p-y)**2)/w.sum())
print('weighted complete',wr(P),'weighted test-like missing',wr(PM))
m=C['masks'].any(1); print('weighted missing-rows only',np.sqrt(np.sum(w[m]*(PM[m]-y[m])**2)/w[m].sum()), 'unweighted',rmse(PM[m],y[m]))
M=C['masks']; names=['t','h','o','p']
pat=np.array([''.join(n for n,b in zip(names,r) if b) or '-' for r in M])
tot=np.sum(w*(PM-y)**2)
for p_ in sorted(set(pat)):
    mm=pat==p_; c=np.sum(w[mm]*(PM[mm]-y[mm])**2)
    print(p_.ljust(4),mm.sum(),'w-rmse',round(np.sqrt(c/w[mm].sum()),2),'share of weighted SSE',round(c/tot,3))
