from __future__ import annotations
import json,random,time,hashlib,math
from pathlib import Path
from collections import defaultdict
import numpy as np,torch
from torch import nn
from models.v2.models import XY,MLPFull,TransformerFull,CTN
from ctn.v2.encoders import StructuredEncoder

YNAMES={"mechanics":["velocity_delta","displacement_delta"],"thermal":["temperature_delta","final_temperature"],"weather":["wetness_delta","final_wetness"]}
YIDX={"mechanics":[0,1],"thermal":[2,3],"weather":[4,5]}; SID={"mechanics":[0,1],"thermal":[2],"weather":[3]}
STATE_FIELDS={"mechanics":["final_velocity","final_position"],"thermal":["final_temperature"],"weather":["final_wetness"]}

def load_v2(data_dir="data/v2/v2.2"):
    root=Path(data_dir); schema=json.loads((root/"feature_schema.json").read_text(encoding="utf-8")); raw={s:[json.loads(x) for x in (root/f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train","validation","test")}
    maxdims={g:max(schema["domains"][d][g]["dimension"] for d in schema["domains"]) for g in ("X","C","S")}; enc={d:{g:StructuredEncoder.from_schema(schema["domains"][d][g]) for g in ("X","C","S")} for d in schema["domains"]}
    stats=json.loads((root/"statistics.json").read_text(encoding="utf-8"))["field_distributions"]
    # state transition normalization is fit only on the train split.
    state_values=[[] for _ in range(4)]
    for r in raw["train"]:
        d=r["domain"]; st=r["target_state"]["objects"]["system"]
        if d=="mechanics": state_values[0].append(float(st["final_velocity"])); state_values[1].append(float(st["final_position"]))
        elif d=="thermal": state_values[2].append(float(st["final_temperature"]))
        else: state_values[3].append(float(st["final_wetness"]))
    state_norm={i:(float(np.mean(v)),float(np.std(v)) if np.std(v)>1e-12 else 1.) for i,v in enumerate(state_values) if v}
    items={}; pairgroups={};
    for split,rows in raw.items():
        xs=[];cs=[];ss=[];domain=[];y=np.zeros((len(rows),6),np.float32);ym=np.zeros_like(y);yc=np.full(len(rows),-1,np.int64);sn=np.zeros((len(rows),4),np.float32);sm=np.zeros_like(sn);sc=np.full(len(rows),-1,np.int64);ood=[];rangeood=[];pairs=[];domains=[]
        for i,r in enumerate(rows):
            d=r["domain"]; vv=[]
            for g in ("X","C","S"):
                vec,_=enc[d][g].transform_one(r[f"{g}_raw"]); vv.append(np.pad(vec,(0,maxdims[g]-len(vec))))
            xs.append(vv[0]);cs.append(vv[1]);ss.append(vv[2]);domain.append([float(d==k) for k in ("mechanics","thermal","weather")]); domains.append(d);ood.append(r["ood_condition"]);pairs.append(r["pair_id"])
            outside=False
            for g in ("X","C","S"):
                for field,value in r[f"{g}_raw"].items():
                    key=f"train.{d}.{g}.{field}"; stat=stats.get(key)
                    if stat and stat["min"] is not None and isinstance(value,(int,float)) and not isinstance(value,bool): outside |= value<stat["min"] or value>stat["max"]
            rangeood.append(bool(outside))
            yraw=r["Y_raw"]; params=schema["targets"]["targets"]
            names=YNAMES[d]
            for ix,name in zip(YIDX[d],names):
                p=params[f"{d}.{name}"]; y[i,ix]=(float(yraw[name])-p["mean"])/p["std"];ym[i,ix]=1
            if d=="thermal": yc[i]=0 if yraw["final_phase"]=="ice" else 1
            st=r["target_state"]["objects"]["system"]
            if d=="mechanics": rawstate=[st["final_velocity"],st["final_position"]]; ids=[0,1]
            elif d=="thermal": rawstate=[st["final_temperature"]];ids=[2];sc[i]=0 if st["final_phase"]=="ice" else 1
            else: rawstate=[st["final_wetness"]];ids=[3]
            for idx,val in zip(ids,rawstate):
                mean,std=state_norm[idx];sn[i,idx]=(float(val)-mean)/std;sm[i,idx]=1
        grp=defaultdict(list)
        for i,pid in enumerate(pairs): grp[pid].append(i)
        pairgroups[split]=[sorted(v,key=lambda i:rows[i]["intervention"]["is_intervention"]) for v in grp.values()]
        items[split]={k:torch.tensor(np.asarray(v),dtype=torch.float32) for k,v in {"x":xs,"c":cs,"s":ss,"d":domain,"y":y,"ym":ym,"sn":sn,"sm":sm}.items()}
        items[split]["yc"]=torch.tensor(yc);items[split]["sc"]=torch.tensor(sc);items[split]["ood"]=np.asarray(ood);items[split]["rangeood"]=np.asarray(rangeood);items[split]["domains"]=domains;items[split]["raw"]=rows
    return items,pairgroups,maxdims,state_norm,schema

def _new(name,dims,ablation=None):
    dx,dc,ds=(dims["X"],dims["C"],dims["S"]) if isinstance(dims,dict) else dims
    if name=="XY": return XY(dx,dc,ds,h=64)
    if name=="MLP-Full": return MLPFull(dx,dc,ds,h=88)
    if name=="Transformer-Full": return TransformerFull(dx,dc,ds,h=32)
    if name.startswith("CTN"):
        ab=ablation or set()
        if "no-C" in name: ab={*ab,"no_c"}
        if "no-S" in name: ab={*ab,"no_s"}
        if "no-XC" in name: ab={*ab,"no_xc"}
        if "no-XS" in name: ab={*ab,"no_xs"}
        if "no-CS" in name: ab={*ab,"no_cs"}
        if "no-transition" in name: ab={*ab,"no_transition"}
        return CTN(dx,dc,ds,h=29,ablation=ab)
    return CTN(dx,dc,ds,h=29,ablation=ablation)

def _loss(out,b,idx,track,pairlist):
    losses={}; terms=[]
    for j in range(6):
        mask=b["ym"][idx,j]>0
        if mask.any():
            z=((out["y_num"][mask,j]-b["y"][idx[mask],j])**2).mean();losses[f"y_num_{j}"]=z;terms.append(z)
    mask=b["yc"][idx]>=0
    if mask.any():
        z=nn.functional.cross_entropy(out["y_phase"][mask],b["yc"][idx[mask]]);losses["y_phase"]=z;terms.append(z)
    for j in range(4):
        mask=b["sm"][idx,j]>0
        if mask.any():
            z=((out["s_num"][mask,j]-b["sn"][idx[mask],j])**2).mean();losses[f"s_num_{j}"]=z;terms.append(z)
    mask=b["sc"][idx]>=0
    if mask.any():
        z=nn.functional.cross_entropy(out["s_phase"][mask],b["sc"][idx[mask]]);losses["s_phase"]=z;terms.append(z)
    total=torch.stack(terms).mean() if terms else torch.tensor(0.)
    if track=="B" and pairlist:
        pair_idx=torch.as_tensor(pairlist,dtype=torch.long,device=out["y_num"].device); left,right=pair_idx[:,0],pair_idx[:,1]
        actual=b["y"][right]-b["y"][left]; predicted=out["y_num"][right]-out["y_num"][left]; mask=b["ym"][left]>0
        pair_chunks=[(predicted-actual).square()[mask]]
        phase_mask=b["yc"][left]>=0
        if phase_mask.any():
            probs=out["y_phase"].softmax(-1); labels=nn.functional.one_hot(b["yc"][pair_idx[phase_mask]],2).float()
            phase_delta=labels[:,1]-labels[:,0]; pred_delta=probs[right[phase_mask]]-probs[left[phase_mask]]
            pair_chunks.append((pred_delta-phase_delta).square().mean(-1))
        nonempty=[v.reshape(-1) for v in pair_chunks if v.numel()]
        if nonempty: losses["pair"]=torch.cat(nonempty).mean();total=total+losses["pair"]
    return total,losses

def _evaluate(model,b,pairs,track,device):
    model.eval(); n=len(b["x"]); out=[]
    t0=time.perf_counter()
    with torch.no_grad():
        for start in range(0,n,2048):
            idx=torch.arange(start,min(n,start+2048)); res=model(*(b[k][idx].to(device) for k in ("x","c","s","d"))); out.append({k:v.cpu() for k,v in res.items()})
    out={k:torch.cat([r[k] for r in out]) for k in out[0]}; idx=torch.arange(n); total,losses=_loss(out,b,idx,"A",[])
    return out,{"losses":{k:float(v) for k,v in losses.items()},"inference_seconds":time.perf_counter()-t0}

def _metrics(out,b,schema,selection=None):
    result={}; domains=b["domains"]; selection=np.ones(len(domains),dtype=bool) if selection is None else np.asarray(selection,dtype=bool)
    for d in ("mechanics","thermal","weather"):
        mask=np.asarray([x==d for x in domains]) & selection; result[d]={}
        for ix,name in zip(YIDX[d],YNAMES[d]):
            ids=np.where(mask & (b["ym"][:,ix].numpy()>0))[0]; pred=out["y_num"][ids,ix].numpy(); truth=b["y"][ids,ix].numpy(); p=schema["targets"]["targets"][f"{d}.{name}"]; rawpred=pred*p["std"]+p["mean"]; rawtruth=truth*p["std"]+p["mean"]
            result[d][name]={"standardized_mse":float(np.mean((pred-truth)**2)) if len(ids) else None,"raw_mae":float(np.mean(np.abs(rawpred-rawtruth))) if len(ids) else None,"count":len(ids)}
        ids=np.where(mask & (b["yc"].numpy()>=0))[0]
        if len(ids):
            pred=out["y_phase"][ids].argmax(1).numpy(); truth=b["yc"][ids].numpy(); result[d]["final_phase"]={"accuracy":float(np.mean(pred==truth)),"confusion_matrix":[[int(np.sum((truth==i)&(pred==j))) for j in range(2)] for i in range(2)],"classes":["ice","liquid"]}
    return result

def _counterfactual_metrics(out,b,pairgroups,schema):
    result={}
    for d in ("mechanics","thermal","weather"):
        rows_by_target={}; direction=[]; invalid=[]
        for pair in pairgroups:
            if len(pair)!=2: continue
            a,z=pair; row=b["raw"][a]
            if row["domain"]!=d: continue
            true_delta={};pred_delta={}
            for ix,name in zip(YIDX[d],YNAMES[d]):
                p=schema["targets"]["targets"][f"{d}.{name}"]
                ta=float(b["y"][a,ix])*p["std"]+p["mean"];tz=float(b["y"][z,ix])*p["std"]+p["mean"]
                pa=float(out["y_num"][a,ix])*p["std"]+p["mean"];pz=float(out["y_num"][z,ix])*p["std"]+p["mean"]
                td=tz-ta;pd=pz-pa;true_delta[name]=td;pred_delta[name]=pd
                rows_by_target.setdefault(name,[]).append({"true_change":td,"predicted_change":pd,"change_mae":abs(td-pd),"absolute_error_before":abs(pa-ta),"absolute_error_after":abs(pz-tz)})
                if abs(td)>1e-10: direction.append(pd*td>0)
            if row["relation_label"]=="NO_EFFECT": invalid.append(max((abs(v) for v in pred_delta.values()),default=0.)<=0.05)
        result[d]={"targets":{k:{"pair_count":len(v),"change_mae":float(np.mean([x['change_mae'] for x in v])),"mean_abs_error_before":float(np.mean([x['absolute_error_before'] for x in v])),"mean_abs_error_after":float(np.mean([x['absolute_error_after'] for x in v]))} for k,v in rows_by_target.items()},"direction_accuracy":float(np.mean(direction)) if direction else None,"direction_pairs":len(direction),"invalid_intervention_preservation_rate":float(np.mean(invalid)) if invalid else None,"invalid_pair_count":len(invalid)}
    return result

def run_suite(data_dir="data/v2/v2.2",output_dir="reports/phase4/v1",smoke=False,seeds=(11,23,37,42,53),epochs=20):
    torch.set_num_threads(2); device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data,pairs,dims,state_norm,schema=load_v2(data_dir); main_names=["XY","MLP-Full","Transformer-Full","CTN"]
    ablation_names=["CTN-no-C","CTN-no-S","CTN-no-XC","CTN-no-XS","CTN-no-CS","CTN-no-transition","CTN-no-pair-loss"]
    if smoke:
        keep=240
        for split in data:
            pairs[split]=pairs[split][:max(2,min(len(pairs[split]),keep))]
            idx=[i for pair in pairs[split] for i in pair]; pairs[split]=[[j for j in range(k*2,k*2+2)] for k in range(len(pairs[split]))]
            for key in ("x","c","s","d","y","ym","sn","sm","yc","sc","ood"): data[split][key]=data[split][key][idx]
            data[split]["rangeood"]=data[split]["rangeood"][idx]; data[split]["domains"]=[data[split]["domains"][i] for i in idx]; data[split]["raw"]=[data[split]["raw"][i] for i in idx]
        seeds=(11,);epochs=2
    records=[]; outdir=Path(output_dir);outdir.mkdir(parents=True,exist_ok=True)
    variants=[("A",""),("B","pair")]
    for track,_ in variants:
      names=main_names+(ablation_names if track=="A" else [])
      for name in names:
       for seed in seeds:
        random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        model=_new(name,dims).to(device); params=sum(p.numel() for p in model.parameters()); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
        tr=data["train"];vl=data["validation"]; tpairs=pairs["train"];vpairs=pairs["validation"]
        ckpt=outdir/"checkpoints"/f"{track}_{name.replace('-','_')}_{seed}.pt"; predfile=outdir/"predictions"/f"{track}_{name.replace('-','_')}_{seed}.jsonl"
        if not smoke and ckpt.exists() and predfile.exists():
            saved=torch.load(ckpt,map_location="cpu",weights_only=False);model.load_state_dict(saved["state_dict"]);model.eval()
            valout,valrun=_evaluate(model,vl,vpairs,"A",device); test=data["test"]
            saved_rows=[json.loads(line) for line in predfile.read_text(encoding="utf-8").splitlines()]
            if len(saved_rows)!=len(test["raw"]): raise RuntimeError(f"Incomplete saved predictions: {predfile}")
            yn=torch.zeros((len(saved_rows),6));yp=torch.zeros((len(saved_rows),2));sn=torch.zeros((len(saved_rows),4));sp=torch.zeros((len(saved_rows),2))
            state_names={"mechanics":[("final_velocity",0),("final_position",1)],"thermal":[("final_temperature",2)],"weather":[("final_wetness",3)]}
            for ri,(pr,row) in enumerate(zip(saved_rows,test["raw"])):
                d=row["domain"]
                for j,target in zip(YIDX[d],YNAMES[d]):
                    p=schema["targets"]["targets"][f"{d}.{target}"];yn[ri,j]=(pr["Y_prediction"][target]-p["mean"])/p["std"]
                if d=="thermal": yp[ri,0 if pr["Y_prediction"]["final_phase"]=="ice" else 1]=1
                for target,j in state_names[d]:
                    mean,std=state_norm[j];sn[ri,j]=(pr["target_state_prediction"][target]-mean)/std
                if d=="thermal": sp[ri,0 if pr["target_state_prediction"]["final_phase"]=="ice" else 1]=1
            testout={"y_num":yn,"y_phase":yp,"s_num":sn,"s_phase":sp}
            rec={"track":track,"model":name,"seed":seed,"parameters":params,"best_epoch":saved["best_epoch"],"train_seconds":None,"inference_seconds":None,"epochs":[],"validation_metrics":_metrics(valout,vl,schema),"test_metrics":_metrics(testout,test,schema),"ood_metrics":_metrics(testout,test,schema,test["ood"]),"out_of_range_metrics":_metrics(testout,test,schema,test["rangeood"]),"counterfactual_metrics":_counterfactual_metrics(testout,test,pairs["test"],schema),"ctn_relation_activations":getattr(model,"activations",{}),"resumed_from_checkpoint":True,"epoch_history_unavailable_after_interruption":True}
            records.append(rec);continue
        best=float("inf");best_ep=0;wait=0;best_state=None;logs=[];start=time.perf_counter()
        for ep in range(1,epochs+1):
            model.train();random.shuffle(tpairs); ep_losses=[]; loss_log=defaultdict(list);grad_norms=[];nonfinite_grad_steps=0
            for st in range(0,len(tpairs),128):
                pg=tpairs[st:st+128]; flat=[i for p in pg for i in p];idx=torch.tensor(flat,dtype=torch.long)
                batch={k:(v.to(device) if isinstance(v,torch.Tensor) else v) for k,v in tr.items() if isinstance(v,torch.Tensor)}
                sliced={k:v[idx.to(device)] for k,v in batch.items()}; out=model(*(sliced[k] for k in ("x","c","s","d"))); pair_idx=[(2*j,2*j+1) for j in range(len(pg))]; local_idx=torch.arange(len(flat),device=device)
                # Track B pair indices are remapped to this batch's flattened row order.
                loss,parts=_loss(out,sliced,local_idx,track,pair_idx if track=="B" else [])
                opt.zero_grad();loss.backward()
                grad_parts=[p.grad.detach().norm(2) for p in model.parameters() if p.grad is not None]
                grad_norm=torch.stack(grad_parts).norm(2) if grad_parts else torch.tensor(0.)
                if not torch.isfinite(grad_norm): nonfinite_grad_steps+=1
                else: grad_norms.append(float(grad_norm))
                opt.step();ep_losses.append(float(loss.detach()))
                for k,v in parts.items(): loss_log[k].append(float(v.detach()))
            _,vlosses=_evaluate(model,vl,vpairs,"A",device); val=float(np.mean(list(vlosses["losses"].values()))) if vlosses["losses"] else 1e9; tl=float(np.mean(ep_losses))
            logs.append({"epoch":ep,"train_loss":tl,"train_by_target":{k:float(np.mean(v)) for k,v in loss_log.items()},"validation_loss":val,"validation_by_target":vlosses["losses"],"gradient_norm_mean":float(np.mean(grad_norms)) if grad_norms else None,"gradient_norm_max":float(np.max(grad_norms)) if grad_norms else None,"nonfinite_gradient_steps":nonfinite_grad_steps})
            if val<best:best=val;best_ep=ep;wait=0;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            else:wait+=1
            if wait>=4:break
        elapsed=time.perf_counter()-start;model.load_state_dict(best_state); valout,valrun=_evaluate(model,vl,vpairs,"A",device);test=data["test"];testout,testrun=_evaluate(model,test,pairs["test"],"A",device)
        ckpt=outdir/"checkpoints";ckpt.mkdir(parents=True,exist_ok=True);torch.save({"state_dict":best_state,"model":name,"track":track,"seed":seed,"parameters":params,"best_epoch":best_ep,"data_version":schema["data_version"],"dims":dims},ckpt/f"{track}_{name.replace('-','_')}_{seed}.pt")
        pred_dir=outdir/"predictions";pred_dir.mkdir(parents=True,exist_ok=True)
        ypred=testout["y_num"].numpy(); phasepred=testout["y_phase"].argmax(-1).numpy(); statepred=testout["s_num"].numpy(); statephase=testout["s_phase"].argmax(-1).numpy()
        with (pred_dir/f"{track}_{name.replace('-','_')}_{seed}.jsonl").open("w",encoding="utf-8") as pf:
            for row_i,row in enumerate(test["raw"]):
                d=row["domain"]; pred={}
                for j,target in zip(YIDX[d],YNAMES[d]):
                    pp=schema["targets"]["targets"][f"{d}.{target}"]; pred[target]=float(ypred[row_i,j]*pp["std"]+pp["mean"])
                if d=="thermal": pred["final_phase"]="ice" if phasepred[row_i]==0 else "liquid"
                pred_state={}
                state_names={"mechanics":[("final_velocity",0),("final_position",1)],"thermal":[("final_temperature",2)],"weather":[("final_wetness",3)]}
                for state_name,state_i in state_names[d]:
                    mean,std=state_norm[state_i]; pred_state[state_name]=float(statepred[row_i,state_i]*std+mean)
                if d=="thermal": pred_state["final_phase"]="ice" if statephase[row_i]==0 else "liquid"
                pf.write(json.dumps({"sample_id":row["sample_id"],"pair_id":row["pair_id"],"domain":d,"track":track,"model":name,"seed":seed,"Y_prediction":pred,"target_state_prediction":pred_state},ensure_ascii=False)+"\n")
        rec={"track":track,"model":name,"seed":seed,"parameters":params,"best_epoch":best_ep,"train_seconds":elapsed,"inference_seconds":testrun["inference_seconds"],"epochs":logs,"validation_metrics":_metrics(valout,vl,schema),"test_metrics":_metrics(testout,test,schema),"ood_metrics":_metrics(testout,test,schema,test["ood"]),"out_of_range_metrics":_metrics(testout,test,schema,test["rangeood"]),"counterfactual_metrics":_counterfactual_metrics(testout,test,pairs["test"],schema),"ctn_relation_activations":getattr(model,"activations",{})}
        records.append(rec)
    result={"data_version":schema["data_version"],"device":str(device),"seeds":list(seeds),"smoke":smoke,"records":records,"state_transition_normalization_train_only":state_norm,"optimizer":"AdamW","learning_rate":1e-3,"batch_pairs":128,"early_stopping_patience":4,"max_epochs":epochs}
    (outdir/"training_curves.json").write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    import hashlib
    root=Path(data_dir); datahash=json.loads((root/"hashes.json").read_text(encoding="utf-8")); hash_checks={}
    for filename,expected in datahash["files"].items():
        file_path=root/filename; digest=hashlib.sha256(file_path.read_bytes()).hexdigest(); hash_checks[filename]={"expected":expected,"actual":digest,"matches":digest==expected}
    def metrics_summary(records_subset):
        grouped=defaultdict(list)
        for r in records_subset:
            for d,targets in r["test_metrics"].items():
                for target,v in targets.items():
                    if v.get("standardized_mse") is not None: grouped[f"{d}.{target}.standardized_mse"].append(v["standardized_mse"])
                    if v.get("raw_mae") is not None: grouped[f"{d}.{target}.raw_mae"].append(v["raw_mae"])
        return {k:{"mean":float(np.mean(v)),"std":float(np.std(v,ddof=1)) if len(v)>1 else 0.,"per_run":v} for k,v in grouped.items()}
    comp={"data_version":schema["data_version"],"tracks":{t:{n:metrics_summary([r for r in records if r["track"]==t and r["model"]==n]) for n in main_names} for t in ("A","B")},"parameter_counts":{r["model"]:r["parameters"] for r in records}}
    (outdir/"model_comparison.json").write_text(json.dumps(comp,ensure_ascii=False,indent=2),encoding="utf-8")
    cf={f"{r['track']}.{r['model']}.seed{r['seed']}":r["counterfactual_metrics"] for r in records if r["model"] in main_names}
    (outdir/"counterfactual_evaluation.json").write_text(json.dumps(cf,ensure_ascii=False,indent=2),encoding="utf-8")
    abl={f"{r['model']}.seed{r['seed']}":r["test_metrics"] for r in records if r["model"] in ablation_names}
    (outdir/"ablation_results.json").write_text(json.dumps(abl,ensure_ascii=False,indent=2),encoding="utf-8")
    repro={"dataset_version":schema["data_version"],"dataset_hashes":datahash,"dataset_hash_verification":hash_checks,"seeds":list(seeds),"device":str(device),"optimizer":"AdamW","learning_rate":1e-3,"batch_pairs":128,"max_epochs":epochs,"early_stopping_patience":4,"test_used_for_selection":False,"all_records":len(records),"prediction_files":len(list((outdir/"predictions").glob("*.jsonl")))}
    (outdir/"reproducibility.json").write_text(json.dumps(repro,ensure_ascii=False,indent=2),encoding="utf-8")
    return result
