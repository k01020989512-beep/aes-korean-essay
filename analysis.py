import json,numpy as np,pandas as pd,matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from common import *
plt.rcParams['font.family']='Malgun Gothic'; plt.rcParams['axes.unicode_minus']=False
tr,te=load(); mean=to_cls((te.h1+te.h2)/2)
P=np.load('pred_klue_rb.npy'); R=np.load('pred_ridge.npy')
bert=P[:,0]; ens=0.8*bert+0.2*R
models={'표면+문자 TF-IDF (1세대)':R,'KLUE-RoBERTa (2세대)':bert,'앙상블(0.8/0.2)':ens}
rows=[dict(model=k,**evaluate(te,v)) for k,v in models.items()]
rows.append(dict(model='사람-사람 (상한)',qwk_avg=qwk(te.h1,te.h2)))
tab=pd.DataFrame(rows); tab.to_csv('table_main.csv',index=False,encoding='utf-8-sig'); print(tab.round(4))
# fig1: main bar
fig,ax=plt.subplots(figsize=(6,3.2)); v=tab.set_index('model').qwk_avg
ax.barh(v.index[::-1],v.values[::-1],color=['#888','#4c72b0','#4c72b0','#c44e52'][::-1]); ax.set_xlabel('QWK (vs 각 채점자 평균)'); 
for i,x in enumerate(v.values[::-1]): ax.text(x+.005,i,f'{x:.3f}',va='center')
ax.set_xlim(0,.9); plt.tight_layout(); plt.savefig('fig_main.png',dpi=200); plt.close()
# fig2: confusion
cm=confusion_matrix(mean,to_cls(bert),labels=[1,2,3,4]); cmn=cm/cm.sum(1,keepdims=True)
fig,ax=plt.subplots(figsize=(4,3.6)); ax.imshow(cmn,cmap='Blues',vmin=0,vmax=1)
for i in range(4):
    for j in range(4): ax.text(j,i,f'{cmn[i,j]:.2f}\n({cm[i,j]})',ha='center',va='center',fontsize=8,color='white' if cmn[i,j]>.5 else 'black')
ax.set_xticks(range(4),[1,2,3,4]); ax.set_yticks(range(4),[1,2,3,4]); ax.set_xlabel('예측'); ax.set_ylabel('정답(채점자 평균 반올림)'); plt.tight_layout(); plt.savefig('fig_confusion.png',dpi=200); plt.close()
# subgroup QWK
sub=[]
for col in ['grade','subject']:
    for g,d in te.groupby(col):
        idx=d.index; sub.append(dict(group=col,value=g,n=len(d),bert=evaluate(d,bert[idx])['qwk_avg'],ridge=evaluate(d,R[idx])['qwk_avg'],human=qwk(d.h1,d.h2)))
sub=pd.DataFrame(sub); sub.to_csv('table_subgroup.csv',index=False,encoding='utf-8-sig'); print(sub.round(3))
# fig3: analytic criteria QWK
a=json.load(open('res_klue_rb.json'))['analytic_qwk']
hv={k:qwk(te[k+'_1'],te[k+'_2']) for k in ANA}
fig,ax=plt.subplots(figsize=(7,3.2)); x=np.arange(len(ANA)); ax.bar(x-.2,[a[k] for k in ANA],.4,label='모델'); ax.bar(x+.2,[hv[k] for k in ANA],.4,label='사람-사람',color='#c44e52')
ax.set_xticks(x,ANA,rotation=30,ha='right'); ax.set_ylabel('QWK'); ax.legend(); plt.tight_layout(); plt.savefig('fig_analytic.png',dpi=200); plt.close()
pd.DataFrame({'criterion':ANA,'model':[a[k] for k in ANA],'human':[hv[k] for k in ANA]}).to_csv('table_analytic.csv',index=False,encoding='utf-8-sig')
# score-distribution / mid-score bias
dist=pd.DataFrame({'human':np.bincount(mean,minlength=5)[1:],'bert':np.bincount(to_cls(bert),minlength=5)[1:]},index=[1,2,3,4]); print(dist)
# bias by length: does model rely on length?
print('corr(pred,len)=',np.corrcoef(bert,te.len_syl)[0,1],' corr(label,len)=',np.corrcoef((te.h1+te.h2)/2,te.len_syl)[0,1])
# dev log fig
lg=pd.DataFrame(json.load(open('res_klue_rb.json'))['dev_log']); print(lg.round(4))
