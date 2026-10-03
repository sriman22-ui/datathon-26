import sys; sys.path.insert(0,'ml_pipeline')
import guard, features as F, pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
raw=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
# step 4 cleaning report
rep={}
rep['humidity>100']=int((raw.humidity>100).sum()); rep['occupancy<0']=int((raw.occupancy<0).sum()); rep['prev<0']=int((raw.previous_usage<0).sum())
rep['type mismatch']=int((raw.building_id.map(F.B2T)!=raw.building_type).sum()); rep['bad day names']=int((~raw.day_of_week.isin(F.DAYS)).sum())
rep['test humidity>100']=int((te.humidity>100).sum()); rep['test type mismatch']=int((te.building_id.map(F.B2T)!=te.building_type).sum())
print(rep)
df=F.clean(raw); print('rows before/after cleaning',len(raw),len(df))
# step 6 split
tr,va,ts=guard.split(df.drop(columns=['id']),target='energy_usage',test_size=0.15,val_size=0.15,seed=42)
print(len(tr),len(va),len(ts))
tr.to_csv('ml_pipeline/split_train.csv',index=False); va.to_csv('ml_pipeline/split_val.csv',index=False); ts.to_csv('ml_pipeline/split_test.csv',index=False)
f,ax=plt.subplots(1,3,figsize=(14,3.5))
for a,c in zip(ax,['energy_usage','temperature','previous_usage']):
    for n,s in [('train',tr),('val',va),('test',ts)]: a.hist(s[c].dropna(),bins=40,density=True,histtype='step',label=n)
    a.set_title(c); a.legend()
guard.fig(4,'split_distributions',f,f"Random 70/15/15 split ({len(tr)}/{len(va)}/{len(ts)} rows): target and key features have the same distribution in all three parts, so validation scores are representative; the 15% internal test is frozen until step 14.")
# step 7: feature width check
X=F.design(tr.dropna()); print('design width',X.shape)
f,ax=plt.subplots(figsize=(7,3.5)); 
pats=te[list(F.NUMS.values())].isna().apply(lambda r:'+'.join([k for k,c in F.NUMS.items() if r[c]]) or 'none',axis=1).value_counts()
ax.bar(pats.index,pats.values); ax.set_yscale('log'); ax.set_title('Missing-value patterns in test.csv (t/h/o/p)'); plt.setp(ax.get_xticklabels(),rotation=45)
guard.fig(7,'test_missing_patterns',f,f"test.csv has {len(pats)} distinct missing patterns; each gets its own handling (impute the missing inputs from the available ones with models fit on complete training rows, then apply the full structural model), instead of one generic NaN rule.")
