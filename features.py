"""Shared feature engineering (step 7) + preprocessing contract (step 8).
The SAME code is pasted into the platform prediction notebook."""
import numpy as np, pandas as pd
DAYS=['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']
BLD=['ADM_A','BUS_A','BUS_B','ENG_A','ENG_B','LEC_A','LIB_A','RES_A','RES_B','SCI_A','SCI_B','SPT_A']
TYP=['Administration','Business','Engineering','LectureHall','Library','Residential','Science','Sports']
B2T={'ADM_A':'Administration','BUS_A':'Business','BUS_B':'Business','ENG_A':'Engineering','ENG_B':'Engineering',
     'LEC_A':'LectureHall','LIB_A':'Library','RES_A':'Residential','RES_B':'Residential','SCI_A':'Science','SCI_B':'Science','SPT_A':'Sports'}
NUMS={'t':'temperature','h':'humidity','o':'occupancy','p':'previous_usage'}
FULL=('t','h','o','p')

def clean(df):
    """Step 4 rules (no rows dropped): types, clip impossible values, derive dow/type."""
    df=df.copy()
    for c in NUMS.values(): df[c]=pd.to_numeric(df[c],errors='coerce')
    df['humidity']=df.humidity.clip(0,100); df['occupancy']=df.occupancy.clip(lower=0)
    df.loc[df.previous_usage<0,'previous_usage']=np.nan
    df['dow']=df.day_of_week.map({d:i for i,d in enumerate(DAYS)}).fillna(0).astype(int)
    df['building_type']=df.building_id.map(B2T).fillna(df.get('building_type'))
    df['hour']=df.hour.astype(int)%24; df['month']=df.month.astype(int)
    return df

def _oh(v,cats):
    v=np.asarray(v); return (v[:,None]==np.asarray(cats)[None,:]).astype(float)

def design(df,avail=FULL,hinge=29.0):
    """Structural design matrix: building/hour/dow/month effects, type x hour profile,
    type x weekend, per-building slopes for the available numeric features, cooling hinge."""
    B=_oh(df.building_id,BLD); T=_oh(df.building_type,TYP)
    H=_oh(df.hour,range(24)); W=_oh(df.dow,range(7)); M=_oh(df.month,range(1,13))
    we=(df.dow.values>=5).astype(float)[:,None]
    x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    parts=[B,H,W,M,x(H,T),x(we,T)]
    cols=[]
    if 't' in avail: t=df.temperature.values; cols.append(t-28)
    if 'o' in avail: cols.append(df.occupancy.values/100)
    if 'p' in avail: cols.append(df.previous_usage.values/50)
    if cols:
        num=np.column_stack(cols); parts+=[num,x(num,B)]
    if 'h' in avail:
        hu=(df.humidity.values[:,None]-78)/5; parts+=[hu,x(hu,T)]
    if 't' in avail and hinge is not None: parts.append(np.maximum(t-hinge,0)[:,None])
    return np.hstack(parts)

def gbm_frame(df,avail):
    X=pd.DataFrame({'hour':df.hour.values,'dow':df.dow.values,'month':df.month.values,
                    'b':pd.Categorical(df.building_id,BLD)})
    for a in avail: X[NUMS[a]]=df[NUMS[a]].values
    return X

# ---- v8 design: group-scaled blocks (scale = inverse penalty strength), tuned on (adversarially weighted) CV ----
SCALE_V8={"B": 2.0, "H": 0.0625, "W": 1.0, "M": 0.25, "TH": 0.25, "Twe": 1.0, "num": 2.0, "Bnum": 1.0, "hu": 1.0, "huT": 0.03125, "hinge": 1.0, "Bwe": 0.08, "o_hb_T": 0.04, "p_hb_T": 0.32, "t_hb": 0.04}
def design_v8(df,hinge=29.0):
    B=_oh(df.building_id,BLD); T=_oh(df.building_type,TYP); H=_oh(df.hour,range(24)); W=_oh(df.dow,range(7)); M=_oh(df.month,range(1,13))
    we=(df.dow.values>=5).astype(float)[:,None]; x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    t=df.temperature.values; o=df.occupancy.values/100; p=df.previous_usage.values/50
    num=np.column_stack([t-28,o,p]); hu=(df.humidity.values[:,None]-78)/5
    hb=np.column_stack([(df.hour.values>=a)&(df.hour.values<a+4) for a in range(0,24,4)]).astype(float)
    bl={'B':B,'H':H,'W':W,'M':M,'TH':x(H,T),'Twe':x(we,T),'num':num,'Bnum':x(num,B),'hu':hu,'huT':x(hu,T),'hinge':np.maximum(t-hinge,0)[:,None],
        'Bwe':x(we,B),'o_hb_T':x(o[:,None]*hb,T),'p_hb_T':x(p[:,None]*hb,T),'t_hb':(t[:,None]-28)*hb,'o2_T':x((o**2)[:,None],T)}
    return np.hstack([bl[k]*SCALE_V8[k] for k in SCALE_V8])
