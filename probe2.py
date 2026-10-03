import sys; sys.path.insert(0,'ml_pipeline')
import features as F, numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
tr=F.clean(pd.read_csv('train.csv')); d=tr.dropna(subset=list(F.NUMS.values())).reset_index(drop=True); y=d.energy_usage.values
oof=np.zeros(len(d)); S=np.zeros(len(d))
for a,b in KFold(5,shuffle=True,random_state=0).split(d):
    oof[b]=Ridge(1).fit(F.design(d.iloc[a]),y[a]).predict(F.design(d.iloc[b]))
    S[b]=Ridge(1).fit(F.design(d.iloc[a],('t','h','o')),y[a]).predict(F.design(d.iloc[b],('t','h','o')))
d['oof']=oof; d['S']=S; d['r']=d.previous_usage/S; d['res']=y-oof
d['rb']=pd.cut(d.r,[0,.5,.7,.85,1.15,1.4,1.8,4])
print(d.groupby('rb').agg(n=('res','size'),rmse=('res',lambda s:np.sqrt((s**2).mean())),bias=('res','mean')).round(3))
# relation: within extreme rows, energy vs prev and S
x=d[(d.r>1.4)|(d.r<0.6)]
from sklearn.linear_model import LinearRegression
lr=LinearRegression().fit(np.c_[x.previous_usage,x.S],x.energy_usage); print('extreme rows: coef prev,S',lr.coef_,lr.intercept_)
x=d[(d.r<=1.4)&(d.r>=0.6)]
lr=LinearRegression().fit(np.c_[x.previous_usage,x.S],x.energy_usage); print('normal rows: coef prev,S',lr.coef_,lr.intercept_)
print(d.groupby('hour').apply(lambda g: pd.Series({'n':len(g),'ext':((g.r>1.4)|(g.r<0.6)).mean()})).T.round(2).to_string())
