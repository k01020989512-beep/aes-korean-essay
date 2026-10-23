import zipfile,glob,json,random,pandas as pd
from common import *
base='D:/Sehoon/자연어처리/25.서술형 글쓰기 평가 데이터/3.개방데이터/1.데이터/'
df=pd.read_csv('essays.csv'); te=df[df.split=='Validation'].reset_index(drop=True)
smp=te.sample(300,random_state=0).reset_index(drop=True); smp['sid']=[f'v{i:03d}' for i in range(300)]
smp[['sid','file','qid','grade','subject','h1','h2']].to_csv('llm/sample_key.csv',index=False,encoding='utf-8-sig')  # labels: NOT given to judges
# rubrics per qid (from label json of any essay of that qid)
want=set(smp.qid); rub={}; fl={}
for f in glob.glob(base+'Validation/02.라벨링데이터/*.zip'):
    z=zipfile.ZipFile(f)
    for n in z.namelist():
        if n.strip('/') in set(smp.file):
            d=json.loads(z.read(n).decode('utf-8')); q=d['essay_question']['id']
            rub.setdefault(q,d['rubric'])
json.dump(rub,open('llm/rubrics.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(len(rub),'rubrics; qids in sample',len(want)); 
# few-shot examples from TRAINING (balanced over score)
tr=df[df.split=='Training'].copy(); tr['m']=to_cls((tr.h1+tr.h2)/2)
ex=pd.concat([tr[tr.m==k].sample(n,random_state=1) for k,n in [(1,2),(2,2),(3,2),(4,2)]])
json.dump([dict(prompt=r.prompt,keyword=r.keyword,text=r.text,score=int(r.m)) for r in ex.itertuples()],open('llm/fewshot_examples.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
# batches (no labels), 6 batches of 50, shared by both conditions
for b in range(6):
    part=smp.iloc[b*50:(b+1)*50]
    json.dump([dict(sid=r.sid,qid=r.qid,grade=r.grade,subject=r.subject,prompt=r.prompt,keyword=r.keyword,text=r.text) for r in part.itertuples()],open(f'llm/batch_{b}.json','w',encoding='utf-8'),ensure_ascii=False,indent=1)
print(smp.grade.value_counts().to_dict(),smp.subject.value_counts().to_dict()); print(((smp.h1+smp.h2)/2).round().value_counts().to_dict())
r=list(rub.values())[0]; print(r['achievement'],'|',r['purpose'],r['type'])
