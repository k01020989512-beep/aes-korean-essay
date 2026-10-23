import json,glob,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from common import *
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
tr,te=load(); dev=tr.sample(frac=0.05,random_state=0).reset_index(drop=True)
tags=sorted(t[5:-4] for t in glob.glob('pred_klue_rb6_s*.npy'))
PT=[np.load(f'pred_{t}.npy') for t in tags]; PD=[np.load(f'dev_{t}.npy') for t in tags]
R=np.load('pred_ridge.npy'); out={}
per=[evaluate(te,p[:,0]) for p in PT]
out['seeds']={k:(float(np.mean([r[k] for r in per])),float(np.std([r[k] for r in per]))) for k in per[0]}
out['per_seed_qwk_avg']=[r['qwk_avg'] for r in per]
ens=np.mean([p[:,0] for p in PT],0); ensd=np.mean([p[:,0] for p in PD],0); out['seed_ensemble']=evaluate(te,ens)
# threshold calibration on dev (cut points maximizing QWK vs each rater), applied to test
def cut(pred,cuts): return 1+np.searchsorted(cuts,pred)
def score(d,pred,cuts): c=cut(pred,cuts); return (qwk(c,d.h1)+qwk(c,d.h2))/2
cuts=np.array([1.5,2.5,3.5])
for _ in range(4):
    for i in range(3):
        grid=np.linspace(cuts[i]-.4,cuts[i]+.4,41); best=max(grid,key=lambda g:score(dev,ensd,np.sort(np.r_[cuts[:i],g,cuts[i+1:]]))); cuts[i]=best; cuts=np.sort(cuts)
c=cut(ens,cuts); out['cuts']=cuts.round(3).tolist(); out['calibrated']=dict(qwk_avg=(qwk(c,te.h1)+qwk(c,te.h2))/2,dist=np.bincount(c,minlength=5)[1:].tolist())
# keyword stacking (fit on dev only)
def kw(d): return d.apply(lambda r:sum(k.strip() in r.text for k in r.keyword.split(',') if k.strip()),axis=1).values
st=Ridge(1).fit(np.c_[ensd,kw(dev)],(dev.h1+dev.h2)/2); out['bert+keyword_stack']=evaluate(te,st.predict(np.c_[ens,kw(te)])); out['stack_coef']=st.coef_.tolist()
# per-criterion (seed-mean)
out['analytic_qwk']={a:float(np.mean([qwk(to_cls(p[:,1+i]),to_cls((te[a+'_1']+te[a+'_2'])/2)) for p in PT])) for i,a in enumerate(ANA)}
out['human_analytic']={a:qwk(te[a+'_1'],te[a+'_2']) for a in ANA}
out['human']=qwk(te.h1,te.h2); out['ridge']=evaluate(te,R); out['ens_ridge']=evaluate(te,0.8*ens+0.2*R)
# learning curve
logs=[json.load(open(f'res_{t}.json'))['dev_log'] for t in tags]
out['dev_curve']=[float(np.mean([l[e]['dev_qwk_avg'] for l in logs])) for e in range(len(logs[0]))]
json.dump(out,open('res_final.json','w'),indent=1,ensure_ascii=False); np.save('pred_final_ens.npy',ens)
print(json.dumps(out,indent=1,ensure_ascii=False))
# figures
fig,ax=plt.subplots(figsize=(5,3)); ax.plot(range(1,len(out['dev_curve'])+1),out['dev_curve'],marker='o'); ax.set_xlabel('에폭'); ax.set_ylabel('dev QWK (3시드 평균)'); plt.tight_layout(); plt.savefig('fig_curve.png',dpi=200); plt.close()
a=out['analytic_qwk']; hv=out['human_analytic']; fig,ax=plt.subplots(figsize=(7,3.2)); x=np.arange(len(ANA)); ax.bar(x-.2,[a[k] for k in ANA],.4,label='모델(3시드 평균)'); ax.bar(x+.2,[hv[k] for k in ANA],.4,label='사람-사람',color='#c44e52')
ax.set_xticks(x,ANA,rotation=30,ha='right'); ax.set_ylabel('QWK'); ax.legend(); plt.tight_layout(); plt.savefig('fig_analytic.png',dpi=200); plt.close()
