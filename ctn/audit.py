"""Phase-two mechanism, data, counterfactual, and training audit utilities.

Audits simulator behavior by intervention. Model architecture is imported unchanged
from training.py; the audit only adds per-epoch diagnostics and evaluates held-out pairs.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .models import XEncoder, YEncoder, CEncoder, SEncoder
from .simulator import MechanicsWorld, ThermalWorld, WeatherWorld
from .training import XYModel, CTNModel, TransformerBaseline, _read, _vec


def _write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=lambda x: x.item() if hasattr(x,"item") else str(x)), encoding="utf-8")


def _mechanics_out(features):
    world=MechanicsWorld(); before=world.reset(position=features.get("position",0.0),velocity=features.get("velocity",0.0),fixed=features.get("fixed",False),mass=features.get("mass",1.0))
    _,after=world.step(force=features.get("force",0.0),friction=features.get("friction",0.0),dt=features.get("dt",1.0))
    p0=before["objects"]["object_001"]["position"]; p1=after["objects"]["object_001"]["position"]
    return {"position_delta":p1-p0,"position":p1,"velocity":after["objects"]["object_001"]["velocity"]}


def _thermal_out(features):
    world=ThermalWorld(); world.reset(temperature=features.get("temperature",20.0),phase=features.get("phase"))
    before=world.observe(); _,after=world.step(heating_power=features.get("heating_power",0.0),cooling_power=features.get("cooling_power",0.0),dt=features.get("dt",1.0))
    b=before["objects"]["water"]; a=after["objects"]["water"]
    return {"temperature_delta":a["temperature"]-b["temperature"],"temperature":a["temperature"],"phase":a["phase"]}


def _weather_out(features):
    world=WeatherWorld(); world.reset(features.get("ground_wetness",0.2))
    before=world.observe(); _,after=world.step(rainfall=features.get("rainfall",0.0),roof_cover=features.get("roof_cover",0.0),drainage=features.get("drainage",0.0))
    w0=before["objects"]["ground_001"]["ground_wetness"]; w1=after["objects"]["ground_001"]["ground_wetness"]
    return {"ground_wetness_delta":w1-w0,"ground_wetness":w1}


def _y_projection(domain, outcome):
    if domain=="mechanics": return {"position_delta":outcome["position_delta"]}
    if domain=="thermal": return {"temperature_delta":outcome["temperature_delta"],"phase":outcome["phase"]}
    return {"ground_wetness_delta":outcome["ground_wetness_delta"]}


def _intervention_row(domain, feature, before, after, expected, why, evaluator):
    y0=_y_projection(domain,evaluator(before)); y1=_y_projection(domain,evaluator(after))
    observed=False
    for k in y0:
        if isinstance(y0[k],(int,float)) and isinstance(y1[k],(int,float)):
            observed |= not math.isclose(float(y0[k]),float(y1[k]),rel_tol=1e-9,abs_tol=1e-9)
        else: observed |= y0[k]!=y1[k]
    delta={}
    for k in y0:
        if isinstance(y0[k],(int,float)) and isinstance(y1[k],(int,float)): delta[k]=round(y1[k]-y0[k],12)
        else: delta[k]=[y0[k],y1[k]] if y0[k]!=y1[k] else None
    return {"domain":domain,"feature":feature,"input_before":before,"input_after":after,"result_before":y0,"result_after":y1,"result_delta":delta,"expected_mechanism":why,"observed_effect":bool(observed),"mechanism_matches":bool(observed)==expected}


def feature_intervention_audit():
    cases=[]
    m={"force":4.0,"friction":0.2,"mass":2.0,"fixed":False,"position":1.0,"velocity":1.0,"dt":1.0}
    for k,v,expect,why in [("force",6.0,True,"外力改变净加速度和位移"),("friction",0.8,True,"摩擦通过速度项改变净加速度"),("mass",4.0,True,"质量改变同一净力下的加速度"),("fixed",True,True,"固定约束抑制位移"),("position",9.0,False,"位置平移不改变本模型的位置增量"),("velocity",2.0,True,"初速度通过摩擦项改变运动"),("dt",2.0,True,"时间步长改变速度/位置更新")]:
        after={**m,k:v}; cases.append(_intervention_row("mechanics",k,m,after,expect,why,_mechanics_out))
    t={"temperature":-3.0,"heating_power":0.2,"cooling_power":0.0,"freezing_point":0.0,"phase":"ice","dt":1.0}
    for k,v,expect,why in [("heating_power",0.8,True,"加热功率改变温度增量"),("cooling_power",0.4,True,"冷却功率改变温度增量"),("temperature",-1.0,True,"起始温度跨越相变阈值，改变物态结果但不改变温度增量"),("phase","liquid",False,"输入 phase 会在 step 中按温度重新派生，初始化矛盾相不保留"),("freezing_point",5.0,False,"该 C 特征未作为 step 参数，模拟器硬编码 0°C")]:
        after={**t,k:v}; cases.append(_intervention_row("thermal",k,t,after,expect,why,_thermal_out))
    w={"rainfall":0.5,"roof_cover":0.2,"drainage":0.1,"ground_wetness":0.4}
    for k,v,expect,why in [("rainfall",0.8,True,"降雨量乘以暴露比例后增加湿度"),("roof_cover",0.8,True,"遮挡比例减少直接降雨贡献"),("drainage",0.3,True,"排水量直接扣减湿度"),("ground_wetness",0.9,True,"提高初始湿度使状态触及上界裁剪，改变本步湿度增量")]:
        after={**w,k:v}; cases.append(_intervention_row("weather",k,w,after,expect,why,_weather_out))
    return {"method":"调用真实 simulator reset/step；每次仅改变一个字段，其他字段固定","cases":cases,"summary":{"total":len(cases),"observed_effect":sum(r["observed_effect"] for r in cases),"mechanism_matches":sum(r["mechanism_matches"] for r in cases),"mismatches":[r for r in cases if not r["mechanism_matches"]]}}


def _canonical(row, include_ids=False):
    obj=row if include_ids else {k:v for k,v in row.items() if k not in {"sample_id","pair_id","event_id"}}
    return json.dumps(obj,sort_keys=True,ensure_ascii=False,separators=(",",":"))


def data_quality_audit(data_dir="data/generated"):
    p=Path(data_dir); splits={n:[json.loads(x) for x in (p/f"{n}.jsonl").read_text(encoding="utf-8").splitlines()] for n in ("train","validation","test")}
    allrows=[(s,r) for s,rows in splits.items() for r in rows]
    full_hash={s:[hashlib.sha256(_canonical(r).encode()).hexdigest() for r in rows] for s,rows in splits.items()}
    id_sets={s:{r["sample_id"] for r in rows} for s,rows in splits.items()}; pair_sets={s:{r["pair_id"] for r in rows} for s,rows in splits.items()}
    duplicate_within={s:len(h)-len(set(h)) for s,h in full_hash.items()}
    cross_dup={f"{a}_vs_{b}":len(set(full_hash[a])&set(full_hash[b])) for a,b in (("train","validation"),("train","test"),("validation","test"))}
    target_specs={"thermal":("temperature_delta",lambda r:r["Y_raw"]["temperature_delta"]/10.0),"mechanics":("position_delta",lambda r:r["Y_raw"]["position_delta"]/10.0),"weather":("ground_wetness_delta",lambda r:r["Y_raw"]["ground_wetness_delta"])}
    domain_stats={}; feature_stats={}
    for domain in ("thermal","mechanics","weather"):
        rows=[r for r in splits["train"] if r["domain"]==domain]; label,fn=target_specs[domain]; vals=[fn(r) for r in rows]
        q1,q3=np.quantile(vals,[.25,.75]); iqr=q3-q1; lo=q1-1.5*iqr; hi=q3+1.5*iqr
        domain_stats[domain]={"target":label,"raw_range":[float(min(r["Y_raw"][label] for r in rows)),float(max(r["Y_raw"][label] for r in rows))],"model_target_range_after_scale":[float(min(vals)),float(max(vals))],"mean_after_scale":float(np.mean(vals)),"std_after_scale":float(np.std(vals)),"normalization":"thermal and mechanics /10; weather unchanged","iqr_outlier_count":sum(v<lo or v>hi for v in vals),"target_exactly_determined_by_X":domain=="thermal"}
        fields=Counter()
        for key in ("X_raw","C_raw","S_raw"):
            for r in rows:
                for k in r[key]: fields[f"{key}.{k}"]+=1
        fs={}
        for key in ("X_raw","C_raw","S_raw"):
            names=sorted({k for r in rows for k in r[key]})
            for name in names:
                vv=[r[key].get(name,"<missing>") for r in rows]; cnt=Counter(map(str,vv))
                fs[f"{key}.{name}"]={"unique_values":len(cnt),"constant":len(cnt)==1,"top_values":cnt.most_common(3)}
        feature_stats[domain]=fs
    combos={}
    for split,rows in splits.items():
        for domain in ("thermal","mechanics","weather"):
            combos[f"{split}.{domain}"]=len({_canonical({"X":r["X_raw"],"C":r["C_raw"]}) for r in rows if r["domain"]==domain})
    overlap={}
    for domain in ("thermal","mechanics","weather"):
        sets={s:{_canonical({"X":r["X_raw"],"C":r["C_raw"],"S":r["S_raw"]}) for r in rows if r["domain"]==domain} for s,rows in splits.items()}
        overlap[domain]={f"{a}_vs_{b}":len(sets[a]&sets[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}
    return {"split_counts":{s:len(r) for s,r in splits.items()},"duplicates":{"within_split":duplicate_within,"cross_split_identical_records":cross_dup,"sample_id_overlap":{f"{a}_vs_{b}":len(id_sets[a]&id_sets[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))},"pair_id_overlap":{f"{a}_vs_{b}":len(pair_sets[a]&pair_sets[b]) for a,b in (("train","validation"),("train","test"),("validation","test"))}},"unique_XCS_combinations":combos,"cross_split_XCS_overlap":overlap,"target_distributions":domain_stats,"feature_cardinality":feature_stats,"encoding_consistency":{"declared_encoders":"separate 32-dimensional stable hashed encoders","experiment_vectorizer":"_vec converts sorted numeric/bool values to at most 8 scalars and maps every string/category to 0; it does not call the declared encoders","string_features_dropped":["mechanics X.action","thermal S.phase"],"effect":"the tested training representation differs from documented X/Y/C/S encoders; categorical event/phase information is absent from models"},"leakage_review":{"model_input_fields":["X_raw numeric projection","C_raw numeric projection","S_raw numeric projection","domain"],"target_fields":["Y_raw domain-specific scalar"],"target_state_and_state_after_used_as_input":False,"sample_ids_or_pair_ids_used_as_input":False,"potential_shortcuts":["thermal temperature_delta is exactly 10*(heating_power-cooling_power), so X alone determines model target; freezing_point C and temperature/phase S do not affect this scalar target","mechanics fixed is always false in generated rows; position is irrelevant to displacement increment; mass/velocity/friction vary","weather target is one-step clipped wetness delta; model can infer most direct effect from rainfall/C and boundary state"],"normalization_warning":"Input fields are not normalized; target output is divided by 10 for thermal/mechanics but not weather. Training uses pooled MSE, so domains have different output variance and loss weight."}}


def counterfactual_cases(count=80, seed=281):
    rng=random.Random(seed); rows=[]
    for i in range(count):
        # Friction intervention: same initial state, force, mass and external context.
        base={"force":0.0,"mass":2.0,"position":rng.uniform(-.5,.5),"velocity":1.0,"fixed":False,"dt":1.0,"friction":0.1}
        changed={**base,"friction":0.8}
        m0=_mechanics_out(base); m1=_mechanics_out(changed)
        rows.append({"pair_id":f"CF_mechanics_{i:03d}","domain":"mechanics","intervened_feature":"friction","input_before":base,"input_after":changed,"y_before":m0["position_delta"]/10,"y_after":m1["position_delta"]/10,"raw_y_before":m0,"raw_y_after":m1})
        # Temperature changes phase; scalar training target (temperature delta) remains unchanged.
        temp0={"temperature":-2.0,"heating_power":0.0,"cooling_power":0.0,"dt":1.0}; temp1={**temp0,"temperature":2.0}
        t0=_thermal_out(temp0); t1=_thermal_out(temp1)
        rows.append({"pair_id":f"CF_thermal_{i:03d}","domain":"thermal","intervened_feature":"temperature","input_before":temp0,"input_after":temp1,"y_before":t0["temperature_delta"]/10,"y_after":t1["temperature_delta"]/10,"raw_y_before":t0,"raw_y_after":t1})
        # Rainfall intervention with all other conditions held fixed.
        w0={"rainfall":0.2,"roof_cover":0.25,"drainage":0.1,"ground_wetness":0.3}; w1={**w0,"rainfall":0.8}
        a0=_weather_out(w0); a1=_weather_out(w1)
        rows.append({"pair_id":f"CF_weather_{i:03d}","domain":"weather","intervened_feature":"rainfall","input_before":w0,"input_after":w1,"y_before":a0["ground_wetness_delta"],"y_after":a1["ground_wetness_delta"],"raw_y_before":a0,"raw_y_after":a1})
    return rows


def _model_features(domain, x, c, s):
    d=[float(domain==k) for k in ("thermal","mechanics","weather")]
    return [torch.tensor([_vec(x)],dtype=torch.float32),torch.tensor([_vec(c)],dtype=torch.float32),torch.tensor([_vec(s)],dtype=torch.float32),torch.tensor([d],dtype=torch.float32)]


def _cf_inputs(row):
    d=row["domain"]; a=row["input_before"]; b=row["input_after"]
    if d=="mechanics":
        def cv(z): return ({"force":z["force"],"action":"push"},{"friction":z["friction"],"fixed":z["fixed"]},{"position":z["position"],"velocity":z["velocity"],"mass":z["mass"],"fixed":z["fixed"]})
    elif d=="thermal":
        def cv(z): return ({"heating_power":z["heating_power"],"cooling_power":z["cooling_power"]},{"freezing_point":0.0},{"temperature":z["temperature"],"phase":""})
    else:
        def cv(z): return ({"rainfall":z["rainfall"]},{"roof_cover":z["roof_cover"],"drainage":z["drainage"]},{"ground_wetness":z["ground_wetness"]})
    return _model_features(d,*cv(a)),_model_features(d,*cv(b))


def run_training_audit(data_dir="data/generated", output_dir="reports/v1", seeds=(11,23,37), epochs=20, batch_size=64, lr=1e-3):
    data=Path(data_dir); out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    train_x,train_y,_,_=_read(data/"train.jsonl"); val_x,val_y,_,_=_read(data/"validation.jsonl"); test_x,test_y,test_domains,_=_read(data/"test.jsonl")
    cf=counterfactual_cases(); architectures={"XY":lambda:XYModel(128),"XYCS":lambda:CTNModel(64),"Transformer":lambda:TransformerBaseline(32)}
    records=[]
    for seed in seeds:
      for name,factory in architectures.items():
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.set_num_threads(2)
        model=factory(); params=sum(p.numel() for p in model.parameters()); opt=torch.optim.AdamW(model.parameters(),lr=lr); loss_fn=nn.MSELoss()
        loader=DataLoader(TensorDataset(*train_x,train_y),batch_size=batch_size,shuffle=True,generator=torch.Generator().manual_seed(seed))
        epochs_log=[]; grad_nonfinite=0; max_grad=0.0; start=time.perf_counter(); best_loss=float("inf"); best_epoch=0
        for epoch in range(1,epochs+1):
            model.train(); tr=[]
            for *features,target in loader:
                opt.zero_grad(); loss=loss_fn(model(*features),target); loss.backward()
                for p in model.parameters():
                    if p.grad is not None:
                        if not torch.isfinite(p.grad).all(): grad_nonfinite+=1
                        else: max_grad=max(max_grad,float(p.grad.detach().abs().max()))
                opt.step(); tr.append(loss.item())
            model.eval()
            with torch.no_grad(): val_loss=float(loss_fn(model(*val_x),val_y).item())
            train_loss=float(np.mean(tr)); epochs_log.append({"epoch":epoch,"train_mse":train_loss,"validation_mse":val_loss})
            if val_loss<best_loss: best_loss=val_loss; best_epoch=epoch
        train_seconds=time.perf_counter()-start
        with torch.no_grad():
            pred=model(*test_x); test_mse=float(loss_fn(pred,test_y).item()); test_mae=float((pred-test_y).abs().mean().item())
            cf_records=[]
            for case in cf:
                xa,xb=_cf_inputs(case); pa=float(model(*xa).item()); pb=float(model(*xb).item()); ya=case["y_before"]; yb=case["y_after"]
                true_delta=yb-ya; pred_delta=pb-pa
                cf_records.append({"pair_id":case["pair_id"],"domain":case["domain"],"intervened_feature":case["intervened_feature"],"true_y_before":ya,"true_y_after":yb,"pred_y_before":pa,"pred_y_after":pb,"error_before":abs(pa-ya),"error_after":abs(pb-yb),"true_change":true_delta,"predicted_change":pred_delta,"change_magnitude_error":abs(pred_delta-true_delta),"direction_correct":None if abs(true_delta)<=1e-9 else pred_delta*true_delta>0,"zero_effect_preserved":abs(pred_delta)<=0.01 if abs(true_delta)<=1e-9 else None,"raw_y_before":case["raw_y_before"],"raw_y_after":case["raw_y_after"]})
        best_val=best_loss; final_val=epochs_log[-1]["validation_mse"]
        records.append({"model":name,"seed":seed,"parameters":params,"inputs":{"XY":["X","domain"],"XYCS":["X","C","S","domain"],"Transformer":["X","C","S","domain"]}[name],"epochs":epochs_log,"best_epoch":best_epoch,"best_validation_mse":best_val,"final_validation_mse":final_val,"overfit_at_end":final_val>best_val*1.05,"gradient_nonfinite_count":grad_nonfinite,"max_abs_gradient":max_grad,"train_seconds":train_seconds,"test_mse":test_mse,"test_mae":test_mae,"counterfactual_pairs":cf_records})
    summary={}
    for name in architectures:
        rs=[r for r in records if r["model"]==name]; vals=[r["test_mse"] for r in rs]
        summary[name]={"test_mse_mean":statistics.mean(vals),"test_mse_stdev":statistics.stdev(vals) if len(vals)>1 else 0.0,"best_epoch_by_seed":[r["best_epoch"] for r in rs],"overfit_runs":sum(r["overfit_at_end"] for r in rs),"gradient_nonfinite_total":sum(r["gradient_nonfinite_count"] for r in rs),"parameter_count":rs[0]["parameters"],"inputs":rs[0]["inputs"]}
    result={"audit_version":out.name,"settings":{"seeds":list(seeds),"epochs":epochs,"learning_rate":lr,"batch_size":batch_size,"test_used_for_training":False},"summary":summary,"runs":records}
    _write_json(out/"training_diagnostics.json",result)
    return result


def write_audit_reports(data_dir="data/generated", output_dir="reports/v1"):
    out=Path(output_dir); _write_json(out/"feature_intervention.json",{"audit_version":out.name,**feature_intervention_audit()}); _write_json(out/"data_quality.json",{"audit_version":out.name,**data_quality_audit(data_dir)})
    _write_json(out/"counterfactual_pairs.json",{"audit_version":out.name,"method":"Same simulator initial state/exogenous settings; one requested feature changed; model predictions populated after run_training_audit.","cases":counterfactual_cases()})


def merge_counterfactual_predictions(output_dir="reports/v1"):
    out=Path(output_dir); pairs=json.loads((out/"counterfactual_pairs.json").read_text(encoding="utf-8")); diag=json.loads((out/"training_diagnostics.json").read_text(encoding="utf-8"))
    indexed={}
    for run in diag["runs"]:
        for c in run["counterfactual_pairs"]:
            c["seed"]=run["seed"]
            c["direction_correct"]=None if abs(c["true_change"])<=1e-9 else c["predicted_change"]*c["true_change"]>0
            c["zero_effect_preserved"]=abs(c["predicted_change"])<=0.01 if abs(c["true_change"])<=1e-9 else None
            indexed.setdefault(c["pair_id"],{}).setdefault(run["model"],[]).append({k:c[k] for k in ("seed","pred_y_before","pred_y_after","error_before","error_after","predicted_change","change_magnitude_error","direction_correct","zero_effect_preserved")})
    _write_json(out/"training_diagnostics.json",diag)
    for pair in pairs["cases"]: pair["model_predictions"]=indexed.get(pair["pair_id"],{})
    summary={}
    for model in ("XY","XYCS","Transformer"):
        summary[model]={}
        for domain in ("mechanics","thermal","weather"):
            vals=[c for run in diag["runs"] if run["model"]==model for c in run["counterfactual_pairs"] if c["domain"]==domain]
            directions=[c["direction_correct"] for c in vals if c["direction_correct"] is not None]
            summary[model][domain]={"pair_count":len(vals),"mean_error_before":float(np.mean([c["error_before"] for c in vals])),"mean_error_after":float(np.mean([c["error_after"] for c in vals])),"mean_change_magnitude_error":float(np.mean([c["change_magnitude_error"] for c in vals])),"direction_accuracy":float(np.mean(directions)) if directions else None,"zero_effect_preservation_accuracy":float(np.mean([c["zero_effect_preserved"] for c in vals if c["zero_effect_preserved"] is not None])) if any(c["zero_effect_preserved"] is not None for c in vals) else None,"mean_true_change":float(np.mean([c["true_change"] for c in vals])),"mean_predicted_change":float(np.mean([c["predicted_change"] for c in vals]))}
    pairs["metric_summary"]=summary; pairs["model_settings"]={"architectures":"unchanged XY, XYCS, Transformer","seeds":diag["settings"]["seeds"],"pairs_per_domain":80,"test_used_for_training":False}
    _write_json(out/"counterfactual_pairs.json",pairs)


def _next_report_version(base="reports"):
    root=Path(base); root.mkdir(parents=True,exist_ok=True); versions=[]
    for p in root.glob("v*"):
        if p.is_dir() and p.name[1:].isdigit(): versions.append(int(p.name[1:]))
    return root/f"v{max(versions,default=0)+1}"


def write_versioned_audit_markdown(output_dir):
    out=Path(output_dir)
    data=json.loads((out/"data_quality.json").read_text(encoding="utf-8")); train=json.loads((out/"training_diagnostics.json").read_text(encoding="utf-8")); cf=json.loads((out/"counterfactual_pairs.json").read_text(encoding="utf-8")); feat=json.loads((out/"feature_intervention.json").read_text(encoding="utf-8"))
    lines=[f"# CTN 第二阶段因果机制审计报告（{out.name}）","",f"审计版本：`{out.name}`。本次生成独立目录，既有报告不会覆盖。","",
      "## 机制干预检查","",f"直接调用 simulator 的单特征干预测试 {feat['summary']['total']} 项，观察到符合预期的结果响应 {feat['summary']['mechanism_matches']} 项；逐项输入/输出见 `feature_intervention.json`。","",
      "## 数据质量","",f"各 split 样本数：{data['split_counts']}。重复样本、跨 split ID/配对 ID 重叠及完整 X/C/S 重叠见 `data_quality.json`。","",
      "## 训练诊断","","| Model | 参数量 | 最佳 epoch（种子序） | 过拟合运行数 | 非有限梯度 | 测试 MSE 均值±标准差 |","|---|---:|---|---:|---:|---:|"]
    for model,stat in train['summary'].items():
      lines.append(f"| {model} | {stat['parameter_count']} | {', '.join(map(str,stat['best_epoch_by_seed']))} | {stat['overfit_runs']}/{len(train['settings']['seeds'])} | {stat['gradient_nonfinite_total']} | {stat['test_mse_mean']:.6f} ± {stat['test_mse_stdev']:.6f} |")
    lines += ["","逐 epoch train/validation loss 在 `training_diagnostics.json`；不删减种子，测试集不参与梯度更新。","","## 反事实结果","","方向准确率按真实目标变化非零的 pair 计算；热学温度干预对本轮温度增量目标是零效应，另报零效应保持率。各模型分领域指标及单 pair 预测见 `counterfactual_pairs.json`。","","## 解释与限制","","XY、XYCS 与 Transformer 使用当前仓库定义的结构不变。本报告只诊断固定模拟环境中的结构化目标，不推断自然语言因果能力。结合 feature intervention 与 data-quality 文件回答：C/S 在不同领域的有效性取决于当前 Y 定义；目标尺度/无效字段/特征表示问题会影响模型比较。","","详细问题清单与逐字段机制说明见本版本的 `feature_intervention.json`、`data_quality.json`、`counterfactual_pairs.json`。"]
    (out/"phase2_audit.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


if __name__=="__main__":
    version_dir=_next_report_version()
    write_audit_reports(output_dir=version_dir)
    run_training_audit(output_dir=version_dir)
    merge_counterfactual_predictions(output_dir=version_dir)
    write_versioned_audit_markdown(version_dir)
    print(f"Audit reports written to {version_dir}")
