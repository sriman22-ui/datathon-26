import sys; sys.path.insert(0,'ml_pipeline')
from fast_eval import *
import guard, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score
res,(P,PM)=evaluate(ridge_fit(hinge=29.0),'FINAL ridge h29+hT')
# baselines OOF on same folds
fl=C['splits'][0][1]; Pl=np.zeros(len(y)); Pg=np.zeros(len(y))
from sklearn.linear_model import LinearRegression
def plainX(df):
    X=pd.DataFrame({f'b_{b}':(df.building_id==b).astype(float) for b in F.BLD})
    for c in ['hour','dow','month','temperature','humidity','occupancy','previous_usage']: X[c]=df[c].values
    return X
for a,b in fl:
    Pl[b]=LinearRegression().fit(plainX(d.iloc[a]),y[a]).predict(plainX(d.iloc[b]))
    Pg[b]=lgb.LGBMRegressor(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1,n_jobs=2).fit(F.gbm_frame(d.iloc[a],F.FULL),y[a]).predict(F.gbm_frame(d.iloc[b],F.FULL))
tab=[]
for n,p in [('mean dummy',np.full(len(y),y.mean())),('plain linear',Pl),('LightGBM',Pg),('structural ridge',P),('structural ridge, test-like missing',PM)]:
    tab.append((n,rmse(p,y),mean_absolute_error(y,p),r2_score(y,p)))
for t in tab: print('%-38s RMSE %.4f MAE %.4f R2 %.4f'%t)
f,ax=plt.subplots(1,3,figsize=(16,4.2))
ax[0].bar([t[0].replace(', ','\n') for t in tab[1:]],[t[1] for t in tab[1:]],color=['#999','#999','#2a7','#2a7']); ax[0].set_ylabel('5-fold RMSE'); ax[0].set_title('Model comparison (mean dummy = %.1f)'%tab[0][1]); plt.setp(ax[0].get_xticklabels(),fontsize=7)
ax[1].scatter(P,y-P,s=3,alpha=.3); ax[1].axhline(0,color='r'); ax[1].set_xlabel('predicted'); ax[1].set_ylabel('residual'); ax[1].set_title('Residuals vs prediction (structural ridge)')
names=['kf','ext_temp','ext_occu','ext_prev']
import json
runs=[json.loads(l) for l in open('ml_pipeline/runs.jsonl')]
x=np.arange(4); ax[2].bar(x-.2,[res[k] for k in names],.4,label='complete rows'); ax[2].bar(x+.2,[res[k+'_miss'] for k in names],.4,label='test-like missing'); ax[2].set_xticks(x); ax[2].set_xticklabels(['random 5-fold','temp tails','occupancy tail','prev tails']); ax[2].legend(); ax[2].set_title('Robustness checks (RMSE)')
guard.fig(12,'model_evaluation',f,f"Structural ridge beats every baseline (RMSE {tab[3][1]:.3f} vs LightGBM {tab[2][1]:.3f}, plain linear {tab[1][1]:.3f}); residuals are centred with spread growing with level (noise ~ proportional to usage); it stays accurate on extrapolation folds and degrades gracefully to {tab[4][1]:.3f} when test-like missingness is injected.")
