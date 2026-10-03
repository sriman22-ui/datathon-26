import sys; sys.path.insert(0,'ml_pipeline')
import guard, json, pandas as pd, numpy as np, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
tr=pd.read_csv('train.csv'); te=pd.read_csv('test.csv')
prof=guard.profile(tr.drop(columns=['id']),target='energy_usage')
print(json.dumps({k:prof[k] for k in ['shape','duplicates','leakage_suspects','traits']},indent=1))
guard.eda_figures(tr.drop(columns=['id']),target='energy_usage')
NUM=['temperature','humidity','occupancy','previous_usage']
# shift figure
f,ax=plt.subplots(1,4,figsize=(16,3.5))
for a,c in zip(ax,NUM):
    a.hist(tr[c].dropna(),bins=40,density=True,alpha=.55,label='train'); a.hist(te[c].dropna(),bins=40,density=True,alpha=.55,label='test'); a.set_title(c); a.legend()
guard.fig(2,'train_vs_test_shift',f,"test.csv is drawn from a wider distribution than train: temperature 99th pct 35.4 vs 32.5 C, occupancy and previous_usage have heavier tails, so models must extrapolate (trees cannot; linear structure can).")
# missing patterns
f,ax=plt.subplots(figsize=(6,3.5))
a=tr[NUM].isna().sum(1).value_counts(normalize=True).sort_index(); b=te[NUM].isna().sum(1).value_counts(normalize=True).sort_index()
x=np.arange(4); ax.bar(x-.2,[a.get(i,0) for i in x],.4,label='train'); ax.bar(x+.2,[b.get(i,0) for i in x],.4,label='test'); ax.set_yscale('log'); ax.set_xlabel('# missing features in row'); ax.legend(); ax.set_title('Missingness per row')
guard.fig(2,'missing_per_row',f,"18% of test rows miss at least one feature (231 miss 2-3) vs 6% in train (almost always 1), so missing-value handling must be learned from complete rows, not from train's few NaN examples.")
# prev vs target
f,ax=plt.subplots(figsize=(5,5)); ax.scatter(tr.previous_usage,tr.energy_usage,s=3,alpha=.3); ax.plot([0,150],[0,150],'r--'); ax.set_xlabel('previous_usage'); ax.set_ylabel('energy_usage')
guard.fig(2,'prev_vs_target',f,"previous_usage correlates 0.95 with the target; it is the previous hour's reading (known at prediction time, not leakage), and the spread around the diagonal is what other features must explain.")
# hourly profile by type
f,ax=plt.subplots(figsize=(8,4))
for t,g in tr.groupby('building_type'): ax.plot(g.groupby('hour').energy_usage.mean(),label=t)
ax.legend(fontsize=7,ncol=2); ax.set_xlabel('hour'); ax.set_ylabel('mean energy'); ax.set_title('Daily profile by building type')
guard.fig(2,'hourly_profile_by_type',f,"Each building type has its own daily shape, so hour must interact with building type rather than enter as one global effect.")
# temperature partial effect
d=tr.dropna()
f,ax=plt.subplots(figsize=(6,4)); bins=pd.cut(d.temperature,np.arange(23,36.5,1)); r=d.energy_usage-0.35*d.previous_usage
ax.plot([i.mid for i in r.groupby(bins).mean().index],r.groupby(bins).mean().values,'o-'); ax.set_xlabel('temperature'); ax.set_ylabel('energy - 0.35*prev (mean)')
guard.fig(2,'temperature_effect',f,"After removing the previous-hour carry-over, energy rises roughly linearly with temperature and gets steeper above ~30-31 C (extra cooling), which matters because test has many more hot hours.")
