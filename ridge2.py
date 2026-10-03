from fw import *
from sklearn.linear_model import Ridge
tr,te=load()
d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
TYP=sorted(tr.building_type.unique())
def oh(v,cats): return pd.get_dummies(pd.Categorical(v,cats),dtype=float).values
def design(df,opt):
    B=oh(df.building_id,BLD); T=oh(df.building_type,TYP)
    H=oh(df.hour,range(24)); W=oh(df.dow,range(7)); M=oh(df.month,range(1,13))
    we=(df.dow.values>=5).astype(float)[:,None]
    G=B if opt.get('g','b')=='b' else T
    Gn=B if opt.get('gn','b')=='b' else T
    def x(A,G): return (G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    t=df.temperature.values; o=df.occupancy.values; p=df.previous_usage.values; hu=df.humidity.values
    num=np.c_[t-28,hu-78,o/100,p/50,np.maximum(t-30,0)]
    parts=[B,H,W,M,num,x(num,Gn),x(H,G),x(we,G)]
    if 'hwe' in opt: parts.append(x(H*we,T))
    if 'hum0' in opt: parts=[B,H,W,M,num,x(num[:,[0,2,3]],Gn),x(H,G),x(we,G)]
    return np.hstack(parts)
def mk(opt,alpha=1.0):
    def f(dtr,ytr,dte):
        m=Ridge(alpha).fit(design(dtr,opt),ytr); return m.predict(design(dte,opt))
    return f
for k,o in {'bb':{},'typeH':{'g':'t'},'typeH typeN':{'g':'t','gn':'t'},'typeH+hwe':{'g':'t','hwe':1},'typeH hum0':{'g':'t','hum0':1}}.items():
  for a in [0.3,3]:
    print(k,a,end=': '); evaluate(mk(o,a),d,y)
