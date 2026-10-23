import numpy as np, pandas as pd
from sklearn.metrics import cohen_kappa_score
def qwk(a,b): return cohen_kappa_score(np.asarray(a).astype(int),np.asarray(b).astype(int),weights='quadratic',labels=[1,2,3,4])
def to_cls(p): return np.clip(np.floor(np.asarray(p)+0.5),1,4).astype(int)
def evaluate(df,pred):
    """QWK of rounded prediction vs each rater (avg) and vs rounded mean label."""
    c=to_cls(pred); mean=(df.h1+df.h2)/2
    return dict(qwk_r1=qwk(c,df.h1),qwk_r2=qwk(c,df.h2),qwk_avg=(qwk(c,df.h1)+qwk(c,df.h2))/2,
                qwk_mean=qwk(c,to_cls(mean)),rmse=float(np.sqrt(((np.asarray(pred)-mean)**2).mean())))
def load():
    df=pd.read_csv('essays.csv'); return df[df.split=='Training'].reset_index(drop=True),df[df.split=='Validation'].reset_index(drop=True)
ANA=['task_1','content_1','content_2','content_3','organization_1','organization_2','expression_1','expression_2']
