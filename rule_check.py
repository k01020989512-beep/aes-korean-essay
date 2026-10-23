import numpy as np,pandas as pd,re
from common import *
d=pd.read_csv('essays.csv'); te=d[d.split=='Validation'].reset_index(drop=True)
def kwcount(r):
    ks=[k.strip() for k in r.keyword.split(',') if k.strip()]; return sum(k in r.text for k in ks)
te['kw']=te.apply(kwcount,axis=1)
h=to_cls((te.content_3_1+te.content_3_2)/2)
# rubric: 0->1, 1->2, 2->3, 3->4, >=4->5 (content_3 scale is 1-5)
rule=np.clip(te.kw+1,1,5)
for r in (1,2):
    print('content_3 rater',r,'rule QWK',round(cohen_kappa_score(rule,te[f'content_3_{r}'],weights='quadratic'),3),'exact',(rule==te[f'content_3_{r}']).mean().round(3))
print('human-human content_3',round(cohen_kappa_score(te.content_3_1,te.content_3_2,weights='quadratic'),3))
print('corr(kw, holistic mean)',np.corrcoef(te.kw,(te.h1+te.h2)/2)[0,1].round(3))
