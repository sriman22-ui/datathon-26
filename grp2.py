exec(open('grp.py').read().split("scale={n:1.0 for n in names}")[0])
def extra(df):
    B=F._oh(df.building_id,F.BLD); T=F._oh(df.building_type,F.TYP); H=F._oh(df.hour,range(24)); we=(df.dow.values>=5).astype(float)[:,None]
    x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
    t=df.temperature.values; o=df.occupancy.values/100; p=df.previous_usage.values/50
    hb=np.column_stack([(df.hour.values>=a)&(df.hour.values<a+4) for a in range(0,24,4)]).astype(float)
    return {'BH':x(H,B),'Bwe':x(we,B),'THwe':x(H*we,T),'o_hb_T':x(o[:,None]*hb,T),'p_hb_T':x(p[:,None]*hb,T),'t_hb':(t[:,None]-28)*hb,'hingeB':x(np.maximum(t-29,0)[:,None],B)}
EX=extra(d); BL.update(EX); names=list(BL)
scale={"B": 2.0, "H": 0.25, "W": 1.0, "M": 0.25, "TH": 0.5, "Twe": 1.0, "num": 2.0, "Bnum": 1.0, "hu": 1.0, "huT": 0.03125, "hinge": 1.0}
for n in EX: scale[n]=0.01
base=evaluate(scale); print('base',base,flush=True); best=0.5*base[0]+0.5*base[1]
for it in range(2):
    for n in names:
        for f in ([4,16,64] if n in EX and scale[n]<=0.01 else [0.5,2]):
            s=dict(scale); s[n]=scale[n]*f; r=evaluate(s); obj=0.5*r[0]+0.5*r[1]
            if obj<best-1e-4: best=obj; scale=s; print(it,n,f,'->',round(r[0],4),round(r[1],4),flush=True)
print(json.dumps(scale)); print('final',evaluate(scale))
