from fw import *
import lightgbm as lgb
tr,te=load(); cols=['hour','dow','month']+NUM
def X(df): x=df[cols].copy(); x['b']=pd.Categorical(df.building_id,BLD); return x
m=lgb.LGBMRegressor(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1).fit(X(tr),tr.energy_usage)
pd.DataFrame({'id':te.id,'prediction':m.predict(X(te))}).to_csv('sub_v0_lgb.csv',index=False)
