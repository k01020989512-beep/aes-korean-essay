import sys,json,time,numpy as np,pandas as pd,torch,torch.nn as nn
from transformers import AutoTokenizer,AutoModel,get_linear_schedule_with_warmup
from common import *
name=sys.argv[1]; tag=sys.argv[2]; EP=int(sys.argv[3]) if len(sys.argv)>3 else 3
SEED=int(sys.argv[4]) if len(sys.argv)>4 else 0
torch.manual_seed(SEED); np.random.seed(SEED)
tr,te=load()
dev=tr.sample(frac=0.05,random_state=0); tr=tr.drop(dev.index).reset_index(drop=True); dev=dev.reset_index(drop=True)
tok=AutoTokenizer.from_pretrained(name)
cols=['h']+ANA
def targets(d):
    t={'h':(d.h1+d.h2)/2}
    for a in ANA: t[a]=(d[a+'_1']+d[a+'_2'])/2
    return torch.tensor(pd.DataFrame(t)[cols].values,dtype=torch.float32)
def enc(d): return tok(list(d.prompt),list(d.text),truncation=True,max_length=320)
class DS(torch.utils.data.Dataset):
    def __init__(s,d): s.e=enc(d); s.y=targets(d)
    def __len__(s): return len(s.y)
    def __getitem__(s,i): return {k:v[i] for k,v in s.e.items()},s.y[i]
def coll(b):
    x=tok.pad([i[0] for i in b],return_tensors='pt'); return x,torch.stack([i[1] for i in b])
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.b=AutoModel.from_pretrained(name); s.h=nn.Linear(s.b.config.hidden_size,len(cols))
    def forward(s,x):
        o=s.b(**{k:v for k,v in x.items() if k!='token_type_ids'}).last_hidden_state
        m=x['attention_mask'].unsqueeze(-1).float(); return s.h((o*m).sum(1)/m.sum(1))+2.9
dl=lambda d,sh,bs:torch.utils.data.DataLoader(DS(d),batch_size=bs,shuffle=sh,collate_fn=coll)
Ltr,Ldev,Lte=dl(tr,True,16),dl(dev,False,64),dl(te,False,64)
m=M().cuda(); opt=torch.optim.AdamW(m.parameters(),lr=3e-5,weight_decay=0.01)
sch=get_linear_schedule_with_warmup(opt,int(0.06*EP*len(Ltr)),EP*len(Ltr)); sc=torch.cuda.amp.GradScaler()
w=torch.tensor([3.]+[1.]*len(ANA)).cuda()
def pred(L):
    m.eval(); o=[]
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        for x,_ in L: o.append(m({k:v.cuda() for k,v in x.items()}).float().cpu())
    return torch.cat(o).numpy()
best=-1; log=[]
for ep in range(EP):
    m.train(); t0=time.time(); tot=0
    for i,(x,y) in enumerate(Ltr):
        with torch.autocast('cuda',dtype=torch.float16):
            p=m({k:v.cuda() for k,v in x.items()})
        loss=(((p.float()-y.cuda())**2)*w).mean()
        opt.zero_grad(); sc.scale(loss).backward(); sc.unscale_(opt); nn.utils.clip_grad_norm_(m.parameters(),1.0); sc.step(opt); sc.update(); sch.step(); tot+=loss.item()
        if i%100==0: print(ep,i,round(tot/(i+1),4),round(time.time()-t0),"s",flush=True)
    pd_=pred(Ldev); r=evaluate(dev,pd_[:,0]); log.append(dict(epoch=ep,train_loss=tot/len(Ltr),**{'dev_'+k:v for k,v in r.items()}))
    print(ep,'dev',r,'time',time.time()-t0,flush=True)
    if r['qwk_avg']>best: best=r['qwk_avg']; torch.save(m.state_dict(),f'{tag}.pt')
m.load_state_dict(torch.load(f'{tag}.pt')); P=pred(Lte); np.save(f'pred_{tag}.npy',P)
res=evaluate(te,P[:,0]); res['dev_log']=log
# per-criterion QWK on test
res['analytic_qwk']={a:qwk(to_cls(P[:,1+i]),to_cls((te[a+'_1']+te[a+'_2'])/2)) for i,a in enumerate(ANA)}
json.dump(res,open(f'res_{tag}.json','w'),indent=1); print('TEST',res)
