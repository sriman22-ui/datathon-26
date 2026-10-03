import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
tr['dow']=tr.day_of_week.map({d:i for i,d in enumerate(D)})
d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
B=pd.get_dummies(d.building_id,dtype=float); H=pd.get_dummies(d.hour,prefix='h',dtype=float); W=pd.get_dummies(d.dow,prefix='w',dtype=float); M=pd.get_dummies(d.month,prefix='m',dtype=float)
def bx(cols): return pd.concat([B.mul(cols[c],axis=0).add_suffix('_'+c) for c in cols],axis=1)
def lin(parts,yy=y):
    X=pd.concat(parts,axis=1).values
    p=cross_val_predict(RidgeCV(alphas=np.logspace(-4,2,20)),X,yy,cv=KFold(5,shuffle=True,random_state=0)); return p
def r(p): return np.sqrt(np.mean((p-y)**2))
t=d.temperature; o=d.occupancy
num=pd.DataFrame({'t':t,'hu':d.humidity,'o':o,'p':d.previous_usage})
core=[B,H,W,M,num,bx(num),bx(H),bx(pd.DataFrame({'we':(d.dow>=5)*1.0})),bx(M)]
print('core',r(lin(core)))
ext=pd.DataFrame({'t2':(t-27.5)**2,'tc':np.maximum(t-26,0),'tc2':np.maximum(t-28,0),'th':np.maximum(24-t,0),'o2':o**2/100,'os':np.sqrt(o),'ol':np.log1p(o)})
print('+ext global',r(lin(core+[ext])))
print('+ext per bldg',r(lin(core+[ext,bx(ext)])))
# multiplicative? log target
p=lin(core+[ext,bx(ext)],np.log(y)); print('log target',r(np.exp(p)))
