exec(open('grp2.py').read().split("scale={\"B\"")[0])
def extra2(df):
    T=F._oh(df.building_type,F.TYP); B=F._oh(df.building_id,F.BLD); H=F._oh(df.hour,range(24)); x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    o=df.occupancy.values/100; p=df.previous_usage.values/50; t=df.temperature.values-28; we=(df.dow.values>=5).astype(float)
    return {'p_H_T':x(p[:,None]*H,T),'o_H_T':x(o[:,None]*H,T),'p_H':p[:,None]*H,'o_we_T':x((o*we)[:,None],T),'p_we_T':x((p*we)[:,None],T),'t_T_hb':x(t[:,None]*np.column_stack([(df.hour.values>=a)&(df.hour.values<a+6) for a in range(0,24,6)]).astype(float),T),'po_T':x((p*o)[:,None],T),'p2_T':x((p**2)[:,None],T),'o2_T':x((o**2)[:,None],T)}
EX2=extra2(d); BL.update(EX2); names=list(BL)
scale={"B": 2.0, "H": 0.125, "W": 1.0, "M": 0.25, "TH": 0.25, "Twe": 1.0, "num": 2.0, "Bnum": 1.0, "hu": 1.0, "huT": 0.03125, "hinge": 1.0, "BH": 0.01, "Bwe": 0.08, "THwe": 0.01, "o_hb_T": 0.04, "p_hb_T": 0.32, "t_hb": 0.04, "hingeB": 0.01}
for n in EX2: scale[n]=0.01
base=evaluate(scale); print('base',base,flush=True); best=0.5*base[0]+0.5*base[1]
for it in range(2):
    for n in names:
        for f in ([4,16,64] if scale[n]<=0.01 else [0.5,2]):
            s=dict(scale); s[n]=scale[n]*f; r=evaluate(s); obj=0.5*r[0]+0.5*r[1]
            if obj<best-1e-4: best=obj; scale=s; print(it,n,f,'->',round(r[0],4),round(r[1],4),flush=True)
print(json.dumps(scale)); print('final',evaluate(scale))
json.dump(scale,open('scale_v8.json','w'))
