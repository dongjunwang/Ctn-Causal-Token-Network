"""Small CPU-friendly comparison: Transformer, XY, and full XYCS."""
from __future__ import annotations

import json, random, time, hashlib
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from .simulator import ThermalWorld, MechanicsWorld, WeatherWorld


def _vec(group):
    vals = [float(v) if isinstance(v, (int, float, bool)) else 0.0 for _, v in sorted(group.items())]
    return (vals[:8] + [0.0] * 8)[:8]


def _read(path):
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]
    domains = {name: i for i, name in enumerate(("thermal", "mechanics", "weather"))}
    features, targets, labels, pair_ids = [], [], [], []
    for row in rows:
        x, c, s = row["X_raw"], row["C_raw"], row["S_raw"]
        domain = row["domain"]
        if domain == "thermal": target = row["Y_raw"]["temperature_delta"] / 10.0
        elif domain == "mechanics": target = row["Y_raw"]["position_delta"] / 10.0
        else: target = row["Y_raw"]["ground_wetness_delta"]
        onehot = [float(domain == d) for d in domains]
        features.append([_vec(x), _vec(c), _vec(s), onehot])
        targets.append([float(target)]); labels.append(domain); pair_ids.append(row["pair_id"])
    tensors = [torch.tensor(np.asarray([f[i] for f in features]), dtype=torch.float32) for i in range(4)]
    return tensors, torch.tensor(targets, dtype=torch.float32), labels, pair_ids


def _chain_examples(paths=30, max_depth=5, seed=971):
    """Roll out held-out simulator trajectories and retain state-conditioned transitions."""
    rng=random.Random(seed); examples=[]
    for domain in ("thermal","mechanics","weather"):
      for path in range(paths):
        d=[float(domain==k) for k in ("thermal","mechanics","weather")]
        if domain=="thermal":
          world=ThermalWorld(); world.reset(rng.uniform(-6,6)); heat,cool=(1.0,0.0) if path%2==0 else (0.0,1.0)
        elif domain=="mechanics":
          world=MechanicsWorld(); world.reset(position=rng.uniform(-1,1),velocity=rng.uniform(-.5,.5),mass=rng.choice([1.,2.])); force,friction=rng.choice([1.,3.]),rng.choice([0.,.2])
        else:
          world=WeatherWorld(); world.reset(rng.uniform(0,.4)); rain,cover,drain=rng.choice([.1,.3]),rng.choice([0.,.5]),rng.choice([0.,.1])
        for depth in range(1,max_depth+1):
          before=world.observe()
          if domain=="thermal":
            state=before["objects"]["water"]; x={"heating_power":heat,"cooling_power":cool}; c={"freezing_point":0.0}; _,after=world.step(heat,cool)
            y=(after["objects"]["water"]["temperature"]-state["temperature"])/10.0
          elif domain=="mechanics":
            state=before["objects"]["object_001"]; x={"force":force,"action":"push"}; c={"friction":friction,"fixed":False}; _,after=world.step(force,friction)
            y=(after["objects"]["object_001"]["position"]-state["position"])/10.0
          else:
            state=before["objects"]["ground_001"]; x={"rainfall":rain}; c={"roof_cover":cover,"drainage":drain}; _,after=world.step(rain,cover,drain)
            y=after["objects"]["ground_001"]["ground_wetness"]-state["ground_wetness"]
          examples.append({"domain":domain,"depth":depth,"x":_vec(x),"c":_vec(c),"s":_vec(state),"d":d,"y":y})
    return examples


class XYModel(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.net = nn.Sequential(nn.Linear(11, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU(), nn.Linear(hidden, 1))
    def forward(self, x, c, s, d): return self.net(torch.cat([x, d], dim=-1))


class CTNModel(nn.Module):
    def __init__(self, hidden=64, include_c=True, include_s=True):
        super().__init__(); self.include_c=include_c; self.include_s=include_s
        self.x=nn.Linear(8,hidden); self.c=nn.Linear(8,hidden) if include_c else None; self.s=nn.Linear(8,hidden) if include_s else None; self.domain=nn.Linear(3,hidden)
        n=2+int(include_c)+int(include_s); self.mix=nn.Sequential(nn.Linear(hidden*n,hidden),nn.GELU(),nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x,c,s,d):
        tokens=[self.x(x)]
        if self.include_c: tokens.append(self.c(c))
        if self.include_s: tokens.append(self.s(s))
        tokens.append(self.domain(d))
        return self.mix(torch.cat(tokens,dim=-1))


class TransformerBaseline(nn.Module):
    def __init__(self, hidden=64):
        super().__init__(); self.proj=nn.ModuleList([nn.Linear(8,hidden),nn.Linear(8,hidden),nn.Linear(8,hidden),nn.Linear(3,hidden)]); self.pos=nn.Parameter(torch.zeros(1,4,hidden)); layer=nn.TransformerEncoderLayer(hidden,4,hidden*2,batch_first=True,dropout=0.0,activation="gelu"); self.encoder=nn.TransformerEncoder(layer,2); self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,x,c,s,d):
        z=torch.stack([p(v) for p,v in zip(self.proj,(x,c,s,d))],dim=1)+self.pos
        return self.head(self.encoder(z)[:,0,:])


def run_experiment(data_dir="data/generated", output_dir="results", seeds=(11,23,37), epochs=15, batch_size=64, lr=1e-3):
    data=Path(data_dir); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    train_x,train_y,_,_=_read(data/"train.jsonl"); val_x,val_y,_,_=_read(data/"validation.jsonl"); test_x,test_y,test_labels,test_pairs=_read(data/"test.jsonl")
    chain_rows=_chain_examples()
    chain_x=[torch.tensor(np.asarray([r[k] for r in chain_rows]),dtype=torch.float32) for k in ("x","c","s","d")]
    chain_y=np.asarray([r["y"] for r in chain_rows]); chain_depth=np.asarray([r["depth"] for r in chain_rows]); chain_domains=np.asarray([r["domain"] for r in chain_rows])
    data_hash=hashlib.sha256(b"".join((data/f"{name}.jsonl").read_bytes() for name in ("train","validation","test"))).hexdigest()
    # 隐藏宽度分别校准，使三种架构参数量落在相近范围。
    architectures={"xy":lambda:XYModel(128),"xyc_ablation":lambda:CTNModel(64,True,False),"xys_ablation":lambda:CTNModel(64,False,True),"xycs":lambda:CTNModel(64),"transformer":lambda:TransformerBaseline(32)}
    records=[]
    for seed in seeds:
      for name,ctor in architectures.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
        model=ctor(); params=sum(p.numel() for p in model.parameters()); opt=torch.optim.AdamW(model.parameters(),lr=lr); loss_fn=nn.MSELoss()
        dl=DataLoader(TensorDataset(*train_x,train_y),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
        start=time.perf_counter()
        epoch_losses=[]
        for _ in range(epochs):
          model.train()
          batch_losses=[]
          for *features,target in dl:
            opt.zero_grad(); loss=loss_fn(model(*features),target); loss.backward(); opt.step(); batch_losses.append(loss.item())
          epoch_losses.append(float(np.mean(batch_losses)))
        train_seconds=time.perf_counter()-start; model.eval()
        with torch.no_grad():
          val_pred=model(*val_x)
          infer_start=time.perf_counter(); test_pred=model(*test_x); inference_seconds=time.perf_counter()-infer_start
          val_mse=loss_fn(val_pred,val_y).item(); test_mse=loss_fn(test_pred,test_y).item(); test_mae=(test_pred-test_y).abs().mean().item()
          errors=(test_pred-test_y).squeeze().tolist()
          tolerance_accuracy=float(((test_pred-test_y).abs() <= 0.05).float().mean().item())
          by_domain={d:float(np.mean([e*e for e,l in zip(errors,test_labels) if l==d])) for d in sorted(set(test_labels))}
          pred_values=test_pred.squeeze().tolist(); true_values=test_y.squeeze().tolist(); paired={}
          for i,pid in enumerate(test_pairs): paired.setdefault(pid,[]).append(i)
          pair_errors=[]; direction_hits=[]
          for ids in paired.values():
            if len(ids)!=2: continue
            a,b=ids; true_delta=true_values[b]-true_values[a]; pred_delta=pred_values[b]-pred_values[a]
            pair_errors.append((pred_delta-true_delta)**2)
            if abs(true_delta)>1e-8: direction_hits.append(float((pred_delta*true_delta)>0))
          pair_mse=float(np.mean(pair_errors)) if pair_errors else None
          direction_accuracy=float(np.mean(direction_hits)) if direction_hits else None
          chain_pred=model(*chain_x).squeeze().tolist()
          chain_metrics={str(dep):float(np.mean([(float(chain_pred[i])-chain_y[i])**2 for i in range(len(chain_rows)) if chain_depth[i]==dep])) for dep in range(1,6)}
          chain_by_domain={dom:float(np.mean([(float(chain_pred[i])-chain_y[i])**2 for i in range(len(chain_rows)) if chain_domains[i]==dom])) for dom in ("thermal","mechanics","weather")}
        records.append({"model":name,"seed":seed,"parameters":params,"epochs":epochs,"learning_rate":lr,"batch_size":batch_size,"final_train_mse":epoch_losses[-1],"validation_mse":val_mse,"validation_mae":float((val_pred-val_y).abs().mean().item()),"test_mse":test_mse,"test_mae":test_mae,"tolerance_accuracy_abs_error_le_0_05":tolerance_accuracy,"counterfactual_pair_mse":pair_mse,"counterfactual_direction_accuracy":direction_accuracy,"test_mse_by_domain":by_domain,"long_chain_mse_by_depth":chain_metrics,"long_chain_mse_by_domain":chain_by_domain,"train_seconds":train_seconds,"inference_seconds":inference_seconds,"data_split":"train/validation/test fixed files"})
    result={"dataset":"generated JSONL","data_sha256":data_hash,"seeds":list(seeds),"epochs":epochs,"learning_rate":lr,"batch_size":batch_size,"records":records,"summary":{}}
    for name in architectures:
      vals=[r["test_mse"] for r in records if r["model"]==name]
      selected=[r for r in records if r["model"]==name]
      result["summary"][name]={"mean_test_mse":float(np.mean(vals)),"std_test_mse":float(np.std(vals,ddof=1)) if len(vals)>1 else 0.0,"mean_counterfactual_pair_mse":float(np.mean([r["counterfactual_pair_mse"] for r in selected])),"mean_counterfactual_direction_accuracy":float(np.mean([r["counterfactual_direction_accuracy"] for r in selected])),"mean_parameters":float(np.mean([r["parameters"] for r in selected])),"mean_long_chain_mse_by_depth":{str(dep):float(np.mean([r["long_chain_mse_by_depth"][str(dep)] for r in selected])) for dep in range(1,6)}}
    (out/"experiment.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
