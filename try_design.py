import sys; sys.path.insert(0,'ml_pipeline')
from fast_eval import *
def ext_design(opts,hinge=29.0):
    def des(df,av=F.FULL):
        X=F.design(df,av,hinge); parts=[X]
        H=F._oh(df.hour,range(24)); T=F._oh(df.building_type,F.TYP); B=F._oh(df.building_id,F.BLD)
        we=(df.dow.values>=5).astype(float)[:,None]
        x=lambda A,G:(G[:,:,None]*A[:,None,:]).reshape(len(df),-1)
        if 'p' in av and 'pH' in opts: parts.append(H*(df.previous_usage.values[:,None]/50))
        if 'p' in av and 'pT' in opts: parts.append(x(H*(df.previous_usage.values[:,None]/50),T)*0.5)
        if 'o' in av and 'oH' in opts: parts.append(H*(df.occupancy.values[:,None]/100))
        if 'o' in av and 'oTwe' in opts: parts.append(x(we*df.occupancy.values[:,None]/100,T))
        if 't' in av and 'tH' in opts: parts.append(H*(df.temperature.values[:,None]-28))
        if 'TwH' in opts: parts.append(x(H*we,T)*0.5)
        if 'BH' in opts: parts.append(x(H,B)*0.5)
        if 'h' in av and 'hT' in opts: parts.append(x((df.humidity.values[:,None]-78)/5,T))
        if 'o' in av and 'o2' in opts: parts.append(x((df.occupancy.values[:,None]/100)**2,T))
        if 't' in av and 'tT2' in opts: parts.append(x(np.maximum(df.temperature.values-29,0)[:,None],T))
        if 'BM' in opts: parts.append(x(F._oh(df.month,range(1,13)),T)*0.5)
        return np.hstack(parts)
    return des
for o in [(),('pH',),('pT',),('oH',),('oTwe',),('tH',),('TwH',),('BH',),('hT',),('o2',),('tT2',),('BM',)]:
    evaluate(ridge_fit(design=ext_design(o)),'h29 +'+'+'.join(o))
