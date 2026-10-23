import sys,json,numpy as np,pandas as pd,torch,torch.nn as nn
from transformers import AutoTokenizer
from common import *
seed=int(sys.argv[1]); torch.manual_seed(seed); np.random.seed(seed)
tr,te=load(); dev=tr.sample(frac=0.05,random_state=0); tr=tr.drop(dev.index).reset_index(drop=True); dev=dev.reset_index(drop=True)
tok=AutoTokenizer.from_pretrained('klue/roberta-base'); V=tok.vocab_size
def enc(d): return tok(list(d.text),truncation=True,max_length=320)['input_ids']
def pad(ids): L=max(map(len,ids)); return torch.tensor([i+[0]*(L-len(i)) for i in ids])
Xtr,Xdev,Xte=enc(tr),enc(dev),enc(te); ytr=torch.tensor(((tr.h1+tr.h2)/2).values,dtype=torch.float32)
class Net(nn.Module):
    def __init__(s):
        super().__init__(); s.e=nn.Embedding(V,200,padding_idx=0); s.c=nn.Conv1d(200,128,5,padding=2); s.l=nn.LSTM(128,128,batch_first=True,bidirectional=True); s.o=nn.Linear(256,1); s.d=nn.Dropout(0.3)
    def forward(s,x):
        m=(x!=0).unsqueeze(-1).float(); h=torch.relu(s.c(s.d(s.e(x)).transpose(1,2))).transpose(1,2); h,_=s.l(h)
        return s.o(s.d((h*m).sum(1)/m.sum(1))).squeeze(-1)+2.9
net=Net().cuda(); opt=torch.optim.Adam(net.parameters(),1e-3)
def pred(X):
    net.eval(); o=[]
    with torch.no_grad():
        for i in range(0,len(X),128): o.append(net(pad(X[i:i+128]).cuda()).cpu())
    return torch.cat(o).numpy()
best=-1
for ep in range(12):
    net.train(); p=np.random.permutation(len(Xtr))
    for i in range(0,len(p),32):
        b=p[i:i+32]; loss=((net(pad([Xtr[j] for j in b]).cuda())-ytr[b].cuda())**2).mean(); opt.zero_grad(); loss.backward(); opt.step()
    r=evaluate(dev,pred(Xdev))['qwk_avg']; print(ep,round(r,4),flush=True)
    if r>best: best=r; torch.save(net.state_dict(),f'cnnlstm_s{seed}.pt')
net.load_state_dict(torch.load(f'cnnlstm_s{seed}.pt')); P=pred(Xte); np.save(f'pred_cnnlstm_s{seed}.npy',P); res=evaluate(te,P); json.dump(res,open(f'res_cnnlstm_s{seed}.json','w')); print('TEST',res)
