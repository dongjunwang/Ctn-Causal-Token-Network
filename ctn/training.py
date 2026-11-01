"""Small CPU-friendly comparison: Transformer, XY, and full XYCS."""
from __future__ import annotations

import json, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


def _read(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    domains = {name: i for i, name in enumerate(("thermal", "mechanics", "weather"))}
    features, targets, labels = [], [], []
    for row in rows:
        x, c, s = row["X_raw"], row["C_raw"], row["S_raw"]
        domain = row["domain"]
        if domain == "thermal": target = row["Y_raw"]["temperature_delta"] / 10.0
        elif domain == "mechanics": target = row["Y_raw"]["position_delta"] / 10.0
        else: target = row["Y_raw"]["ground_wetness"]
        def vec(group):
            return np.asarray([float(v) if isinstance(v, (int, float, bool)) else 0.0 for _, v in sorted(group.items())], dtype=np.float32)[:8].tolist() + [0.0] * max(0, 8-len(group))
        onehot = [float(domain == d) for d in domains]
        features.append([vec(x), vec(c), vec(s), onehot])
        targets.append([float(target)]); labels.append(domain)
    tensors = [torch.tensor(np.asarray([f[i] for f in features]), dtype=torch.float32) for i in range(4)]
    return tensors, torch.tensor(targets, dtype=torch.float32), labels


class XYModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(11, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x, c, s, d): return self.net(torch.cat([x, d], dim=-1))


class CTNModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.x=nn.Linear(8,hidden); self.c=nn.Linear(8,hidden); self.s=nn.Linear(8,hidden); self.domain=nn.Linear(3,hidden); self.mix=nn.Sequential(nn.Linear(hidden*4,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x,c,s,d): return self.mix(torch.cat([self.x(x),self.c(c),self.s(s),self.domain(d)],dim=-1))


class TransformerBaseline(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.proj=nn.ModuleList([nn.Linear(8,hidden),nn.Linear(8,hidden),nn.Linear(8,hidden),nn.Linear(3,hidden)]); self.pos=nn.Parameter(torch.zeros(1,4,hidden)); layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,batch_first=True,dropout=0.0,activation="gelu"); self.encoder=nn.TransformerEncoder(layer,2); self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x,c,s,d):
        z=torch.stack([p(v) for p,v in zip(self.proj,(x,c,s,d))],dim=1)+self.pos
        return self.head(self.encoder(z)[:,0,:])


def run_experiment(data_dir="data/generated", output_dir="results", seeds=(11,23,37), epochs=15, batch_size=64, lr=1e-3):
    data=Path(data_dir); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    train_x,train_y,_=_read(data/"train.jsonl"); val_x,val_y,_=_read(data/"validation.jsonl"); test_x,test_y,test_labels=_read(data/"test.jsonl")
    # 隐藏宽度分别校准，使三种架构参数量落在相近范围。
    architectures={"xy":lambda:XYModel(128),"xycs":lambda:CTNModel(64),"transformer":lambda:TransformerBaseline(32)}
    records=[]
    for seed in seeds:
      for name,ctor in architectures.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
        model=ctor(); params=sum(p.numel() for p in model.parameters()); opt=torch.optim.AdamW(model.parameters(),lr=lr); loss_fn=nn.MSELoss()
        dl=DataLoader(TensorDataset(*train_x,train_y),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
        start=time.perf_counter()
        for _ in range(epochs):
          model.train()
          for *features,target in dl:
            opt.zero_grad(); loss=loss_fn(model(*features),target); loss.backward(); opt.step()
        train_seconds=time.perf_counter()-start; model.eval()
        with torch.no_grad():
          val_pred=model(*val_x)
          infer_start=time.perf_counter(); test_pred=model(*test_x); inference_seconds=time.perf_counter()-infer_start
          val_mse=loss_fn(val_pred,val_y).item(); test_mse=loss_fn(test_pred,test_y).item(); test_mae=(test_pred-test_y).abs().mean().item()
          errors=(test_pred-test_y).squeeze().tolist()
          by_domain={d:float(np.mean([e*e for e,l in zip(errors,test_labels) if l==d])) for d in sorted(set(test_labels))}
        records.append({"model":name,"seed":seed,"parameters":params,"epochs":epochs,"learning_rate":lr,"batch_size":batch_size,"validation_mse":val_mse,"test_mse":test_mse,"test_mae":test_mae,"test_mse_by_domain":by_domain,"train_seconds":train_seconds,"inference_seconds":inference_seconds})
    result={"dataset":"generated JSONL","seeds":list(seeds),"records":records,"summary":{}}
    for name in architectures:
      vals=[r["test_mse"] for r in records if r["model"]==name]
      result["summary"][name]={"mean_test_mse":float(np.mean(vals)),"std_test_mse":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,"mean_parameters":float(np.mean([r["parameters"] for r in records if r["model"]==name]))}
    (out/"experiment.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
