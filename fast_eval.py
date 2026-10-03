import sys; sys.path.insert(0,'ml_pipeline')
from model_lib import *
import pickle, json
C=pickle.load(open('ml_pipeline/imp_cache.pkl','rb')); d=C['d']; y=d.energy_usage.values
def evaluate(fit, name, log=True):
    """fit(dtr,ytr) -> predict(df_imputed, df_masked) returning predictions"""
    res={}
    for sname,fl in C['splits']:
        n=len(d); P=np.full(n,np.nan); PM=np.full(n,np.nan)
        for fi,(a,b) in enumerate(fl):
            pr=fit(d.iloc[a],y[a]); dm,imp=C['imp'][(sname,fi)]
            P[b]=pr(d.iloc[b],d.iloc[b]); PM[b]=pr(imp,dm)
        k=~np.isnan(P); res[sname]=round(rmse(P[k],y[k]),4); res[sname+'_miss']=round(rmse(PM[k],y[k]),4)
        if sname=='kf': res['_oof']=(P,PM)
    oof=res.pop('_oof'); res['name']=name
    if log:
        print(json.dumps(res)); open('ml_pipeline/runs.jsonl','a').write(json.dumps(res)+'\n')
    return res,oof
def ridge_fit(alpha=1.0,hinge=30.0,design=None,w=None,p_direct=False):
    des=design or (lambda df,av=F.FULL: F.design(df,av,hinge))
    def fit(dtr,ytr):
        m=Ridge(alpha).fit(des(dtr),ytr)
        dirs={}
        def pr(imp,dm):
            out=m.predict(des(imp))
            if p_direct:
                miss=dm[list(F.NUMS.values())].isna().values; keys=[tuple(r) for r in miss]
                for pat in set(keys):
                    if not pat[3]: continue
                    av=tuple(x for x,mm in zip(F.FULL,pat) if not mm)
                    if av not in dirs: dirs[av]=Ridge(alpha).fit(des(dtr,av),ytr)
                    rows=[i for i,k in enumerate(keys) if k==pat]
                    pdir=dirs[av].predict(des(dm.iloc[rows],av))
                    out[rows]= pdir if p_direct=='direct' else 0.5*(out[rows]+pdir)
            return out
        return pr
    return fit
if __name__=='__main__':
    evaluate(ridge_fit(),'ridge base')
    for a in [0.1,0.3,3,10]: evaluate(ridge_fit(alpha=a),f'ridge a={a}')
    for h in [29.,29.5,30.5,31.,None]: evaluate(ridge_fit(hinge=h),f'ridge hinge={h}')
    evaluate(ridge_fit(p_direct='direct'),'p direct'); evaluate(ridge_fit(p_direct='blend'),'p blend')
