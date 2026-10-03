import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd, itertools, json
from sklearn.model_selection import KFold
tr=pd.read_csv('train_advw.csv'); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values; w=d.w.values
def blocks(df):
    B=F._oh(df.building_id,F.BLD); T=F._oh(df.building_type,F.TYP); H=F._oh(df.hour,range(24)); W=F._oh(df.dow,range(7)); M=F._oh(df.month,range(1,13))
    we=(df.dow.values>=5).astype(float)[:,None]; x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    t=df.temperature.values; num=np.column_stack([t-28,df.occupancy.values/100,df.previous_usage.values/50]); hu=(df.humidity.values[:,None]-78)/5
    return {'B':B,'H':H,'W':W,'M':M,'TH':x(H,T),'Twe':x(we,T),'num':num,'Bnum':x(num,B),'hu':hu,'huT':x(hu,T),'hinge':np.maximum(t-29,0)[:,None]}
BL=blocks(d); names=list(BL)
folds=[list(KFold(5,shuffle=True,random_state=s).split(d)) for s in (0,1)]
# precompute fold gram pieces for fast ridge: solve (X'X + I) b = X'y with column scaling s
def evaluate(scale):
    X=np.hstack([BL[n]*scale[n] for n in names]); X1=np.hstack([X,np.ones((len(X),1))])
    P=np.zeros((2,len(y)))
    for k,fl in enumerate(folds):
        for a,b in fl:
            A=X1[a]; G=A.T@A; R=np.eye(G.shape[0]); R[-1,-1]=0
            beta=np.linalg.solve(G+R,A.T@y[a]); P[k,b]=X1[b]@beta
    r=np.sqrt(np.mean((P-y)**2)); wr=np.sqrt(np.sum(w*(P-y)**2)/(2*w.sum()))
    return r,wr
scale={n:1.0 for n in names}
base=evaluate(scale); print('base',base,flush=True)
best=base[0]*0.5+base[1]*0.5
for it in range(3):
    for n in names:
        for f in [0.25,0.5,2,4]:
            s=dict(scale); s[n]=scale[n]*f; r=evaluate(s); obj=0.5*r[0]+0.5*r[1]
            if obj<best-1e-4: best=obj; scale=s; print(it,n,f,'->',round(r[0],4),round(r[1],4),flush=True)
print(json.dumps(scale)); print('final',evaluate(scale))
