"""Deterministic synthetic classification training; no downloaded weights or external data."""
import argparse, json, pathlib, time
import torch
from torch import nn

def train(out='artifacts/model.pt',seed=42):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    n=12000
    x=torch.randn(n,32)
    coef=torch.randn(32,4)
    labels=(x@coef+0.5*torch.randn(n,4)).argmax(1)
    split=10000
    model=nn.Sequential(nn.Linear(32,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,4))
    optim=torch.optim.Adam(model.parameters(),lr=0.005)
    t=time.perf_counter()
    for epoch in range(15):
        model.train()
        for indices in torch.randperm(split).split(256):
            loss=nn.functional.cross_entropy(model(x[indices]),labels[indices]);optim.zero_grad();loss.backward();optim.step()
    model.eval()
    with torch.inference_mode():
        accuracy=(model(x[split:]).argmax(1)==labels[split:]).float().mean().item()
    p=pathlib.Path(out);p.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'state_dict':model.state_dict(),'features':32,'classes':4,'seed':seed},p)
    report={'data':'synthetic teacher-generated tabular classification','train_examples':split,'test_examples':n-split,'accuracy':accuracy,'epochs':15,'training_seconds':time.perf_counter()-t}
    p.with_suffix('.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))
    return report
if __name__=='__main__':
    pa=argparse.ArgumentParser();pa.add_argument('--output',default='artifacts/model.pt');a=pa.parse_args();train(a.output)
