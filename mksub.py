from pipe import *
import sys
tr,te=load(); d=tr.dropna().reset_index(drop=True); y=d.energy_usage.values
# include rows with missing as training for imputers? main ridge needs complete rows
m=Model(mode='imp').fit(d,y); p=m.predict(te)
pd.DataFrame({'id':te.id,'prediction':p}).to_csv(sys.argv[1],index=False)
print(pd.Series(p).describe())
