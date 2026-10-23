import sys,numpy as np,pandas as pd,torch,torch.nn as nn
from transformers import AutoTokenizer,AutoModel
from common import *
name='klue/roberta-base'; tr,te=load(); dev=tr.sample(frac=0.05,random_state=0).reset_index(drop=True)
tok=AutoTokenizer.from_pretrained(name); cols=['h']+ANA
class M(nn.Module):
    def __init__(s):
        super().__init__(); s.b=AutoModel.from_pretrained(name); s.h=nn.Linear(s.b.config.hidden_size,len(cols))
    def forward(s,x):
        o=s.b(**{k:v for k,v in x.items() if k!='token_type_ids'}).last_hidden_state
        m=x['attention_mask'].unsqueeze(-1).float(); return s.h((o*m).sum(1)/m.sum(1))+2.9
for tag in sys.argv[1:]:
    m=M().cuda(); m.load_state_dict(torch.load(f'{tag}.pt')); m.eval(); out=[]
    with torch.no_grad(), torch.autocast('cuda',dtype=torch.float16):
        for i in range(0,len(dev),64):
            e=dev.iloc[i:i+64]; x=tok(list(e.prompt),list(e.text),truncation=True,max_length=320,padding=True,return_tensors='pt')
            out.append(m({k:v.cuda() for k,v in x.items()}).float().cpu())
    np.save(f'dev_{tag}.npy',torch.cat(out).numpy()); print(tag,'dev saved',evaluate(dev,torch.cat(out).numpy()[:,0]))
