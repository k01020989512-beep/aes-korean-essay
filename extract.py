import zipfile,glob,json,sys,pandas as pd
base='D:/Sehoon/자연어처리/25.서술형 글쓰기 평가 데이터/3.개방데이터/1.데이터/'
rows=[]
for split in ['Training','Validation']:
    for f in sorted(glob.glob(base+f'{split}/02.라벨링데이터/*.zip')):
        z=zipfile.ZipFile(f)
        for n in z.namelist():
            if not n.endswith('.json'): continue
            d=json.loads(z.read(n).decode('utf-8'))
            q,a,s=d['essay_question'],d['essay_answer'],d['score']['personal']
            r=dict(split=split,file=n.strip('/'),qid=q['id'],subject=q['subject'],topic=q['topic'],level=q['level'],
                   grade=q['grade'],prompt=q['prompt'],keyword=q['keyword'].strip(),text=a['text'],
                   len_syl=a['len_syllable'],len_word=a['len_word'])
            h=s['holistic']['score']; r['h1'],r['h2']=h[0],h[1]
            for k,v in s['analytic'].items():
                r[k+'_1'],r[k+'_2']=v['score'][0],v['score'][1]
            rows.append(r)
df=pd.DataFrame(rows); df.to_csv('essays.csv',index=False,encoding='utf-8-sig')
print(df.shape); print(df.split.value_counts()); print(df.grade.value_counts().sort_index()); print(df.subject.value_counts())
print(df[['h1','h2']].describe()); print(pd.crosstab(df.h1,df.h2))
print(df.len_syl.describe()); print(df.isna().sum().sum())
