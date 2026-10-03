import pandas as pd, numpy as np, warnings; warnings.filterwarnings('ignore')
from sklearn.linear_model import LinearRegression
tr=pd.read_csv('train.csv')
D=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
tr['dow']=tr.day_of_week.map({d:i for i,d in enumerate(D)})
d=tr.dropna().reset_index(drop=True)
d['we']=(d.dow>=5)*1.
rows=[]
for b,g in d.groupby('building_id'):
    X=pd.concat([g[['previous_usage','occupancy','temperature','humidity','we']],pd.get_dummies(g.hour,prefix='h',drop_first=True,dtype=float)],axis=1)
    m=LinearRegression().fit(X,g.energy_usage); res=g.energy_usage-m.predict(X)
    rows.append([b,len(g)]+list(m.coef_[:5].round(3))+[round(m.intercept_,1),round(res.std(),2)])
print(pd.DataFrame(rows,columns=['b','n','prev','occ','temp','hum','we','int','resstd']).to_string())
