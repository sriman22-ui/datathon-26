from fw import *
from sklearn.linear_model import Ridge
tr,te=load()
d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
def design(df,opt):
    B=pd.get_dummies(pd.Categorical(df.building_id,BLD),dtype=float).values
    H=pd.get_dummies(pd.Categorical(df.hour,range(24)),dtype=float).values
    W=pd.get_dummies(pd.Categorical(df.dow,range(7)),dtype=float).values
    M=pd.get_dummies(pd.Categorical(df.month,range(1,13)),dtype=float).values
    we=(df.dow.values>=5).astype(float)[:,None]
    def bx(A): return (B[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    t=df.temperature.values; o=df.occupancy.values; p=df.previous_usage.values; hu=df.humidity.values
    num=np.c_[t-28,hu-78,o/100,p/50]
    parts=[B,H,W,M,num,bx(num),bx(H)]
    if 'we' in opt: parts.append(bx(we))
    if 'wd' in opt: parts.append(bx(W))
    if 'bm' in opt: parts.append(bx(M))
    for h in opt.get('hinge',[]): 
        z=np.maximum(t-h,0)[:,None]; parts+= [z, bx(z)] if 'hingeB' in opt else [z]
    if 't2' in opt: parts.append(bx(((t-28)**2)[:,None]/4))
    if 'oh' in opt: parts.append(bx(we*o[:,None]/100)); 
    if 'ohr' in opt:
        hb=np.c_[(df.hour.values<6),(df.hour.values>=6)&(df.hour.values<12),(df.hour.values>=12)&(df.hour.values<18)].astype(float)
        parts.append(bx(hb*o[:,None]/100))
    if 'ph' in opt: parts.append(H*p[:,None]/50)
    if 'th' in opt: parts.append(H*(t-28)[:,None])
    return np.hstack(parts)
def mk(opt,alpha=1.0):
    def f(dtr,ytr,dte):
        m=Ridge(alpha).fit(design(dtr,opt),ytr); return m.predict(design(dte,opt))
    return f
cfgs={'core':{},'we':{'we':1},'we+bm':{'we':1,'bm':1},'wd':{'wd':1},
 'we+h31':{'we':1,'hinge':[31]},'we+h30':{'we':1,'hinge':[30]},'we+h30B':{'we':1,'hinge':[30],'hingeB':1},'we+h31B':{'we':1,'hinge':[31],'hingeB':1},
 'we+t2':{'we':1,'t2':1},'we+h31+oh':{'we':1,'hinge':[31],'oh':1},'we+h31+ohr':{'we':1,'hinge':[31],'ohr':1},
 'we+h31+ph':{'we':1,'hinge':[31],'ph':1},'we+h31+th':{'we':1,'hinge':[31],'th':1}}
for k,o in cfgs.items():
    print(k,end=': '); evaluate(mk(o),d,y)
