from fw import *
from sklearn.linear_model import Ridge
TYP=['Administration','Business','Engineering','LectureHall','Library','Residential','Science','Sports']
def oh(v,cats): return pd.get_dummies(pd.Categorical(v,cats),dtype=float).values
def design(df,opt=None,avail=('t','h','o','p')):
    opt=opt or {}
    B=oh(df.building_id,BLD); T=oh(df.building_type,TYP)
    H=oh(df.hour,range(24)); W=oh(df.dow,range(7)); M=oh(df.month,range(1,13))
    we=(df.dow.values>=5).astype(float)[:,None]
    G=T if opt.get('g','t')=='t' else B
    def x(A,G): return (G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    cols=[]
    if 't' in avail: t=df.temperature.values; cols+=[t-28]; 
    if 'o' in avail: cols+=[df.occupancy.values/100]
    if 'p' in avail: cols+=[df.previous_usage.values/50]
    parts=[B,H,W,M,x(H,G),x(we,G)]
    if cols:
        num=np.column_stack(cols); parts+=[num,x(num,B)]
    if 'h' in avail and opt.get('hum',1): parts.append((df.humidity.values[:,None]-78)/5)
    if 't' in avail and opt.get('hinge',30): parts.append(np.maximum(t-opt.get('hinge',30),0)[:,None])
    for extra in opt.get('extra',[]): parts.append(extra(df,x,B,T,H,W,M,we))
    return np.hstack(parts)
def ridge_fp(opt=None,alpha=1.0,avail=('t','h','o','p')):
    def f(dtr,ytr,dte):
        m=Ridge(alpha).fit(design(dtr,opt,avail),ytr); return m.predict(design(dte,opt,avail))
    return f
