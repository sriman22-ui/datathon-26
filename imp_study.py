import sys; sys.path.insert(0,'ml_pipeline')
from model_lib import *
from sklearn.model_selection import cross_val_predict
tr=pd.read_csv('ml_pipeline/split_train.csv'); va=pd.read_csv('ml_pipeline/split_val.csv')
d=pd.concat([tr,va]).reset_index(drop=True).dropna().reset_index(drop=True)
def rdes(df,av):
    B=F._oh(df.building_id,F.BLD);H=F._oh(df.hour,range(24));W=F._oh(df.dow,range(7));M=F._oh(df.month,range(1,13))
    x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1); we=(df.dow.values>=5).astype(float)[:,None]
    parts=[B,H,W,M,x(H,B),x(we,B),x(H*we,F._oh(df.building_type,F.TYP))]
    if av:
        num=np.column_stack([df[F.NUMS[a]].values for a in av]); parts+=[num,x(num,B)]
    return np.hstack(parts)
kf=KFold(5,shuffle=True,random_state=0)
for tgt in ['o','p','t','h']:
    av=tuple(a for a in F.FULL if a!=tgt); yv=d[F.NUMS[tgt]].values
    g=np.zeros(len(d)); r=np.zeros(len(d))
    for a,b in kf.split(d):
        g[b]=lgb.LGBMRegressor(**{**GP,'n_jobs':2}).fit(F.gbm_frame(d.iloc[a],av),yv[a]).predict(F.gbm_frame(d.iloc[b],av))
        r[b]=Ridge(1.0).fit(rdes(d.iloc[a],av),yv[a]).predict(rdes(d.iloc[b],av))
    print(tgt,'gbm',rmse(g,yv),'ridge',rmse(r,yv),'avg',rmse((g+r)/2,yv), 'sd',yv.std())
