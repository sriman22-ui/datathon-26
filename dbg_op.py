import sys; sys.path.insert(0,'ml_pipeline')
from fast_eval import *
out=np.load('missexp.npy'); base,impw,dl,dlw,rl=out
M=C['masks']; names='thop'
pat=np.array([''.join(n for n,b in zip(names,r) if b) for r in M])
for p_ in ['op','p','o','tp','to','top','hop','thp']:
    m=pat==p_
    if m.sum()==0: continue
    print(p_.ljust(4),m.sum(),'imp',round(rmse(base[m],y[m]),2),'direct_lgb',round(rmse(dl[m],y[m]),2),'resid_lgb(direct ridge+lgb)',round(rmse(rl[m],y[m]),2))
