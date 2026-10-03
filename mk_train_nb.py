import json
feat=open('ml_pipeline/features.py').read().split('"""',2)[2].strip()
art=open('ml_pipeline/artifact.py').read().split('"""',2)[2].strip().replace("import sys; sys.path.insert(0,'ml_pipeline')\n","").replace("import features as F\n","").replace("F.","")
cells=[]
md=lambda s: cells.append({"cell_type":"markdown","metadata":{},"source":s})
code=lambda s: cells.append({"cell_type":"code","metadata":{},"execution_count":None,"outputs":[],"source":s})
md("""# Smart Campus energy forecasting — training notebook (Track 1)

Full pipeline: problem understanding, EDA, cleaning, features, baselines, model comparison, evaluation, error analysis, final training, and saving `model.pkl`. The **prediction notebook** (uploaded to the platform) only loads `model.pkl` and predicts.

**Problem.** Predict hourly `energy_usage` of a campus building from building, time (hour, weekday, month), weather, occupancy and previous-hour usage. Metric: RMSE (MAE, R² also reported). Some inputs are missing.

**Key insights that drive the design**
1. Usage is close to *linear and additive per building*: a carry-over of ~0.35 x previous-hour usage, a type-specific daily profile, per-building slopes for occupancy and temperature, and a steeper cooling effect above ~29-31 °C.
2. `test.csv` comes from a **wider distribution** than train (heavier tails in temperature, humidity, occupancy, previous usage; more heat-wave / storm / transition-hour rows). Trees cannot extrapolate; a well-specified linear model can.
3. Test has **far more missing inputs** (18% of rows, up to 3 per row, vs 6% in train) and test's feature distribution is shifted, so missing values are imputed per availability pattern with models that also see test's (unlabeled) features.""")
code("""import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib, itertools, lightgbm as lgb
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score
train = pd.read_csv('train.csv'); test = pd.read_csv('test.csv')
print(train.shape, test.shape); print(train.head().to_string())""")
md("## 1. Exploratory data analysis")
code("""NUMC=['temperature','humidity','occupancy','previous_usage']
print('missing per column (train/test):'); print(pd.DataFrame({'train':train[NUMC].isna().mean(),'test':test[NUMC].isna().mean()}).round(3))
print('rows with >=1 missing: train %.3f, test %.3f' % (train[NUMC].isna().any(axis=1).mean(), test[NUMC].isna().any(axis=1).mean()))
fig,ax=plt.subplots(1,4,figsize=(18,3.5))
for a,c in zip(ax,NUMC):
    a.hist(train[c].dropna(),bins=40,density=True,alpha=.55,label='train'); a.hist(test[c].dropna(),bins=40,density=True,alpha=.55,label='test'); a.set_title(c); a.legend()
plt.suptitle('Train vs test feature distributions: test has heavier tails / extra edge-case clusters'); plt.show()""")
code("""fig,ax=plt.subplots(1,3,figsize=(18,4))
ax[0].scatter(train.previous_usage,train.energy_usage,s=3,alpha=.3); ax[0].plot([0,150],[0,150],'r--'); ax[0].set_xlabel('previous_usage'); ax[0].set_ylabel('energy_usage'); ax[0].set_title('corr %.2f'%train[['previous_usage','energy_usage']].corr().iloc[0,1])
for t,g in train.groupby('building_type'): ax[1].plot(g.groupby('hour').energy_usage.mean(),label=t)
ax[1].legend(fontsize=7,ncol=2); ax[1].set_title('Daily profile by building type'); ax[1].set_xlabel('hour')
d=train.dropna(); r=d.energy_usage-0.35*d.previous_usage; b=pd.cut(d.temperature,np.arange(23,36.5,1))
ax[2].plot([i.mid for i in r.groupby(b,observed=True).mean().index],r.groupby(b,observed=True).mean().values,'o-'); ax[2].set_title('Temperature effect (after prev carry-over): steeper when hot'); ax[2].set_xlabel('temperature')
plt.show()""")
md("## 2. Cleaning and feature engineering\nNo impossible values were found (humidity <= 100, occupancy >= 0, previous usage >= 0, building type consistent with building id), so no rows are dropped; the rules stay in `clean()` as guards. `design()` builds the structural design matrix.")
code(feat)
md("## 3. Validation design\n5-fold CV, plus **extrapolation folds** (train on the middle of a feature's range, score on its tails) and **test-like missingness** checks.")
code("""tr=clean(train); te=clean(test)
d=tr.dropna(subset=NUMC).reset_index(drop=True); y=d.energy_usage.values
def rmse(a,b): return float(np.sqrt(np.mean((np.asarray(a)-np.asarray(b))**2)))
folds=list(KFold(5,shuffle=True,random_state=0).split(d))
def cv(fitpred):
    P=np.zeros(len(d))
    for a,b in folds: P[b]=fitpred(d.iloc[a],y[a],d.iloc[b])
    return P
def ext_fold(fitpred,col,lo,hi):
    ql,qh=d[col].quantile([lo,hi]); t=((d[col]<ql)|(d[col]>qh)).values if lo>0 else (d[col]>qh).values
    return rmse(fitpred(d[~t],y[~t],d[t]),y[t])""")
md("## 4. Baselines and model comparison")
code("""def plainX(df):
    X=pd.DataFrame({f'b_{b}':(df.building_id==b).astype(float) for b in BLD})
    for c in ['hour','dow','month']+NUMC: X[c]=df[c].values
    return X
models={
 'mean dummy': lambda A,ya,B: np.full(len(B),ya.mean()),
 'plain linear': lambda A,ya,B: LinearRegression().fit(plainX(A),ya).predict(plainX(B)),
 'LightGBM': lambda A,ya,B: lgb.LGBMRegressor(n_estimators=1600,learning_rate=0.02,num_leaves=15,subsample=0.8,subsample_freq=1,colsample_bytree=0.8,verbose=-1).fit(gbm_frame(A,FULL),ya).predict(gbm_frame(B,FULL)),
 'structural ridge': lambda A,ya,B: Ridge(1.0).fit(design(A),ya).predict(design(B)),
}
table={}
for name,fn in models.items():
    P=cv(fn)
    entry={'CV RMSE':rmse(P,y),'MAE':mean_absolute_error(y,P),'R2':r2_score(y,P)}
    for col,lo_q,hi_q,label in [('temperature',.06,.94,'temp tails'),('occupancy',0,.9,'occupancy tail'),('previous_usage',.06,.94,'prev tails')]:
        entry[label]=ext_fold(fn,col,lo_q,hi_q)
    table[name]=entry
results=pd.DataFrame.from_dict(table,orient='index'); print(results.round(3).to_string())""")
md("The structural ridge wins on random folds **and** stays accurate on the extrapolation folds where LightGBM degrades sharply (it flattens outside the training range). Boosting on ridge residuals, linear-tree LightGBM, splines and 20+ interaction variants were also tried (see `ml_pipeline/runs.jsonl`): none improved, i.e. the remaining error (~3.0) is noise.")
md("## 5. Missing inputs\nFor each availability pattern, the missing inputs are imputed by LightGBM trained on the available inputs (train rows **plus unlabeled test features**, weight 10, LightGBM+XGBoost average — test's feature distribution is shifted, e.g. occupancy imputed from train alone is biased by -5.9 on test's own observed values), then the full ridge is applied. Rows missing both occupancy and previous usage use the average of that and a NaN-native LightGBM. Small additive offsets per missing feature are learned out-of-fold on train's real missing rows.")
code(art)
md("## 6. Final training, error analysis and saving the model")
code("""oof=np.zeros(len(train))
for a,b in KFold(5,shuffle=True,random_state=0).split(train):
    m=build(train.iloc[a],unlabeled=test,c_mode='blend'); oof[b]=predict(m,train.iloc[b])
yt=train.energy_usage.values; miss=train[NUMC].isna().any(axis=1).values
print('OOF on all train rows: RMSE %.3f  MAE %.3f  R2 %.4f' % (rmse(oof,yt),mean_absolute_error(yt,oof),r2_score(yt,oof)))
print('complete rows RMSE %.3f | rows with missing inputs RMSE %.3f' % (rmse(oof[~miss],yt[~miss]),rmse(oof[miss],yt[miss])))
r=yt-oof; fig,ax=plt.subplots(1,2,figsize=(14,4))
ax[0].scatter(oof,r,s=3,alpha=.3); ax[0].axhline(0,color='r'); ax[0].set_xlabel('prediction'); ax[0].set_ylabel('residual'); ax[0].set_title('Residuals: centred, spread grows with usage')
pd.Series(r).groupby(train.building_id).apply(lambda s: np.sqrt((s**2).mean())).plot.bar(ax=ax[1]); ax[1].set_title('RMSE by building'); plt.show()""")
code("""final=build(train,unlabeled=test,c_mode='blend')
joblib.dump(final,'model.pkl',compress=('xz',9))
preds=predict(final,test)
sub=pd.DataFrame({'id':test['id'].values,'prediction':preds}); sub.to_csv('submission.csv',index=False); print(sub.head().to_string())""")
md("""## 7. Assumptions, limitations
* **Assumptions:** effects are additive and linear within building (validated: residual boosting finds no structure); noise is roughly proportional to usage; the test-generating process for complete rows matches train's; previous_usage is the true previous-hour reading (legitimately available at prediction time).
* **Limitations:** rows with several missing inputs are much harder (no train analogue); imputation uses unlabeled test features (transductive) — if the private set's feature distribution differs, imputers fall back to train+test statistics; irreducible noise ~3.0 RMSE.
* **With more time/data:** timestamps would allow true lag features; sensor-failure flags would explain test's structured missingness.""")
nb={"nbformat":4,"nbformat_minor":5,"metadata":{"kernelspec":{"name":"python3","display_name":"Python 3","language":"python"},"language_info":{"name":"python"}},"cells":cells}
json.dump(nb,open('deliver/training_notebook.ipynb','w'),indent=1)
