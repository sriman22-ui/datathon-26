import pandas as pd, numpy as np, lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
def fe(df):
    X=df.copy()
    X['dow']=X.day_of_week.map({d:i for i,d in enumerate(D)})
    for c in ['building_id','building_type']: X[c]=X[c].astype('category')
    return X.drop(columns=['id','day_of_week','energy_usage'],errors='ignore')
X=fe(tr); y=tr.energy_usage.values
kf=KFold(5,shuffle=True,random_state=0); oof=np.zeros(len(y))
for a,b in kf.split(X):
    m=lgb.LGBMRegressor(n_estimators=3000,learning_rate=0.02,num_leaves=15,min_child_samples=20,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1)
    m.fit(X.iloc[a],y[a],eval_set=[(X.iloc[b],y[b])],callbacks=[lgb.early_stopping(200,verbose=False)])
    oof[b]=m.predict(X.iloc[b]); print(m.best_iteration_)
r=lambda p:np.sqrt(np.mean((p-y)**2))
print('lgb',r(oof))
p=tr.previous_usage.fillna(tr.energy_usage.mean()); print('prev only',r(p))
mask=tr.previous_usage.notna()
print('lgb rmse prev present',np.sqrt(np.mean((oof-y)[mask]**2)),'missing',np.sqrt(np.mean((oof-y)[~mask]**2)))
