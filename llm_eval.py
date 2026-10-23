import json,glob,numpy as np,pandas as pd
from common import *
key=pd.read_csv('llm/sample_key.csv'); full=pd.read_csv('essays.csv'); te=full[full.split=='Validation'].reset_index(drop=True)
idx=te.reset_index().set_index('file')['index']; key['ti']=key.file.map(idx)
def load(tag):
    d=[x for b in range(6) for x in json.load(open(f'llm/out_{tag}_{b}.json',encoding='utf-8'))]
    s=pd.DataFrame(d).drop_duplicates('sid').set_index('sid').score; assert len(s)==300 and s.between(1,4).all(); return key.sid.map(s).values
P={'LLM zero-shot':load('zs'),'LLM few-shot':load('fs')}
def ev(pred): return evaluate(key,pred)
rows=[dict(model=k,**ev(v)) for k,v in P.items()]
r=np.load('pred_ridge.npy'); rows.append(dict(model='1세대 릿지',**ev(r[key.ti.values])))
import sys; tag=sys.argv[1] if len(sys.argv)>1 else 'klue_rb'
b=np.load(f'pred_{tag}.npy'); b=b[:,0] if b.ndim>1 else b; rows.append(dict(model='KLUE-RoBERTa',**ev(b[key.ti.values])))
rows.append(dict(model='사람-사람',qwk_avg=qwk(key.h1,key.h2)))
t=pd.DataFrame(rows); t.to_csv('table_llm300.csv',index=False,encoding='utf-8-sig'); print(t.round(3).to_string())
mean=to_cls((key.h1+key.h2)/2)
for k,v in P.items():
    print(k,'pred dist',np.bincount(v,minlength=5)[1:],'human',np.bincount(mean,minlength=5)[1:],'exact agree',(v==mean).mean().round(3),'adjacent',(abs(v-mean)<=1).mean().round(3),'corr w/ len',np.corrcoef(v,te.len_syl[key.ti])[0,1].round(3))
print('zs-fs agree',(P['LLM zero-shot']==P['LLM few-shot']).mean().round(3),'qwk',round(qwk(P['LLM zero-shot'],P['LLM few-shot']),3))
# bootstrap CI for qwk_avg
rng=np.random.default_rng(0); 
def ci(pred):
    v=[evaluate(key.iloc[i],pred[i])['qwk_avg'] for i in (rng.integers(0,300,300) for _ in range(500))]; return np.percentile(v,[2.5,97.5]).round(3)
for k,v in {**P,'KLUE-RoBERTa':b[key.ti.values]}.items(): print(k,'95% CI',ci(v))
