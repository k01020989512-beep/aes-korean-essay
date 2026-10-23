import numpy as np,pandas as pd,json,re
from sklearn.linear_model import Ridge
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack,csr_matrix
from common import *
tr,te=load()
def feats(d):
    t=d.text
    sents=t.apply(lambda s:max(1,len(re.findall(r'[.!?]',s))))
    kw=[sum(k.strip() in x for k in kws.split(',')) for x,kws in zip(t,d.keyword)]
    words=t.str.split()
    f=pd.DataFrame(dict(syl=np.log1p(d.len_syl),word=np.log1p(d.len_word),sent=np.log1p(sents),wps=d.len_word/sents,
        ttr=words.apply(lambda w:len(set(w))/max(1,len(w))),kw=kw,kwr=np.array(kw)/d.keyword.str.count(',').add(1),
        comma=t.str.count(','),dblspace=t.str.count('  '),avgw=d.len_syl/d.len_word))
    return f
mu,sd=None,None
Ftr,Fte=feats(tr),feats(te); mu,sd=Ftr.mean(),Ftr.std()
Ftr,Fte=(Ftr-mu)/sd,(Fte-mu)/sd
y=(tr.h1+tr.h2)/2
res={}
m=Ridge(alpha=1).fit(Ftr,y); res['surface_ridge']=evaluate(te,m.predict(Fte))
tf=TfidfVectorizer(analyzer='char_wb',ngram_range=(1,3),min_df=5,sublinear_tf=True,max_features=200000)
Xtr=hstack([tf.fit_transform(tr.text),csr_matrix(Ftr.values)]).tocsr(); Xte=hstack([tf.transform(te.text),csr_matrix(Fte.values)]).tocsr()
m=Ridge(alpha=3).fit(Xtr,y); p=m.predict(Xte); res['surface+char_tfidf_ridge']=evaluate(te,p)
np.save('pred_ridge.npy',p)
res['human_vs_human']=dict(qwk=qwk(te.h1,te.h2))
res['length_only']=evaluate(te,Ridge(1).fit(Ftr[['syl']],y).predict(Fte[['syl']]))
json.dump(res,open('res_baseline.json','w'),indent=1)
for k,v in res.items(): print(k,{a:round(b,4) for a,b in v.items()})
