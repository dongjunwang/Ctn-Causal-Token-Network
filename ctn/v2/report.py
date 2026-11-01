from __future__ import annotations
import json,shutil
from pathlib import Path
from .simulators import MechanicsV2,ThermalV2,WeatherV2

def _case(domain,feature,before,after,sim,ykeys,expect,why):
    a=sim.step(**before); b=sim.step(**after); delta={k:(b[k]-a[k] if isinstance(a[k],(int,float)) and isinstance(b[k],(int,float)) else [a[k],b[k]] if a[k]!=b[k] else None) for k in ykeys}; changed=any(v not in (0,None,[a[k],a[k]]) for k,v in delta.items())
    # Categorical output changes count as an effect; numerics use a tolerance.
    changed=any((abs(v)>1e-10 if isinstance(v,(int,float)) else v is not None) for v in delta.values())
    return {"domain":domain,"feature":feature,"input_before":before,"input_after":after,"Y_before":{k:a[k] for k in ykeys},"Y_after":{k:b[k] for k in ykeys},"Y_delta":delta,"expected_effect":expect,"observed_effect":changed,"mechanism_matches":changed==expect,"mechanism":why}

def intervention_suite():
    cases=[]
    m=MechanicsV2(); b=dict(force=2.,force_direction=1,duration=1.,friction=.2,fixed=False,mass=2.,velocity=1.,position=3.)
    for k,v,e,w in [("force",3.,True,"signed force changes acceleration"),("force_direction",-1,True,"direction changes force sign"),("duration",2.,True,"duration affects velocity/displacement updates"),("friction",.8,True,"friction opposes motion"),("fixed",True,True,"fixed constraint zeros displacement and final velocity"),("mass",4.,True,"mass changes acceleration"),("velocity",2.,True,"initial velocity changes result"),("position",103.,False,"translation does not alter displacement or velocity change")]:
        cases.append(_case("mechanics",k,b,{**b,k:v},m,["velocity_delta","displacement_delta"],e,w))
    t=ThermalV2(); b=dict(heating_power=1.,cooling_power=.1,duration=1.,freezing_point=0.,ambient_temperature=0.,initial_temperature=-.5,initial_phase="ice")
    for k,v,e,w in [("heating_power",2.,True,"heating contributes energy"),("cooling_power",.5,True,"cooling removes energy"),("duration",2.,True,"duration scales heat exchange"),("freezing_point",2.,True,"threshold may change phase but not temperature trajectory"),("ambient_temperature",10.,True,"ambient exchange changes energy balance"),("initial_temperature",1.,True,"initial temperature affects ambient exchange and final phase"),("initial_phase","liquid",True,"phase-dependent heat capacity changes temperature response")]:
        # Freezing threshold case is chosen to cross phase between the values.
        if k=="freezing_point": b={**b,"initial_temperature":.2,"heating_power":0.,"cooling_power":.1,"ambient_temperature":.2}
        cases.append(_case("thermal",k,b,{**b,k:v},t,["temperature_delta","final_temperature","final_phase"],e,w))
    w=WeatherV2(); b=dict(rainfall=.3,rainfall_duration=1.,roof_cover=.2,drainage=.05,ground_wetness=.4)
    for k,v,e,why in [("rainfall",.5,True,"rainfall is multiplied by exposure and duration"),("rainfall_duration",2.,True,"duration scales rain and drainage contributions"),("roof_cover",.8,True,"cover attenuates direct rain; invalid if rainfall is zero"),("drainage",.2,True,"drainage removes wetness"),("ground_wetness",.95,True,"near upper boundary, clipping makes state affect delta")]:
        cases.append(_case("weather",k,b,{**b,k:v},w,["wetness_delta","final_wetness"],e,why))
    invalid={**b,"rainfall":0.}; cases.append(_case("weather","roof_cover (zero-rain invalid intervention)",invalid,{**invalid,"roof_cover":1.},w,["wetness_delta","final_wetness"],False,"with zero rain cover cannot change direct rain contribution"))
    return {"method":"direct one-field-at-a-time calls to V2 simulator step()","total":len(cases),"passed":sum(c["mechanism_matches"] for c in cases),"cases":cases}

def dataset_integrity(data_dir):
    data=Path(data_dir); splits={s:[json.loads(x) for x in (data/f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train","validation","test")}
    ids={s:{r["sample_id"] for r in rs} for s,rs in splits.items()}; pairs={s:{r["pair_id"] for r in rs} for s,rs in splits.items()}; pair_check={}; bad=[]
    for split,rows in splits.items():
        groups={}
        for r in rows: groups.setdefault(r["pair_id"],[]).append(r)
        pair_check[split]={"pair_count":len(groups),"all_have_two_records":all(len(g)==2 for g in groups.values()),"one_intervention_each":all(sum(r["intervention"]["is_intervention"] for r in g)==1 for g in groups.values())}
        for pid,g in groups.items():
            if len(g)!=2: bad.append({"pair_id":pid,"error":"not exactly two records"}); continue
            a,b=sorted(g,key=lambda r:r["intervention"]["is_intervention"])
            changed=[]
            for group in ("X_raw","C_raw","S_raw"):
                for key in set(a[group])|set(b[group]):
                    if a[group].get(key)!=b[group].get(key): changed.append((group,key))
            designated=b["intervention"]["feature"]
            if len(changed)!=1 or changed[0][1]!=designated: bad.append({"pair_id":pid,"changed":changed,"designated":designated})
    overlap={f"{a}_vs_{b}":{"sample_ids":len(ids[a]&ids[b]),"pair_ids":len(pairs[a]&pairs[b])} for a,b in (("train","validation"),("train","test"),("validation","test"))}
    def signature(r): return json.dumps([r["domain"],r["X_raw"],r["C_raw"],r["S_raw"]],sort_keys=True,separators=(",",":"))
    train_factual={signature(r) for r in splits["train"] if not r["intervention"]["is_intervention"]}
    test_ood_factual={signature(r) for r in splits["test"] if r["ood_condition"] and not r["intervention"]["is_intervention"]}
    hashes=json.loads((data/"hashes.json").read_text(encoding="utf-8")); hash_valid={name:__import__('hashlib').sha256((data/name).read_bytes()).hexdigest()==digest for name,digest in hashes["files"].items()}
    return {"split_counts":{s:len(rows) for s,rows in splits.items()},"pair_checks":pair_check,"cross_split_id_overlap":overlap,"paired_feature_change_errors":bad,"test_ood_factual_XCS_overlap_with_train":len(train_factual&test_ood_factual),"sha256_valid":hash_valid,"passed":not bad and all(v["sample_ids"]==0 and v["pair_ids"]==0 for v in overlap.values()) and len(train_factual&test_ood_factual)==0 and all(hash_valid.values())}

def write_phase3_reports(data_dir="data/v2/v2.2",reports_root="reports"):
    data=Path(data_dir); root=Path(reports_root); root.mkdir(parents=True,exist_ok=True)
    versions=[int(p.name.split("_v")[-1]) for p in root.glob("phase3_v*") if p.is_dir() and p.name.split("_v")[-1].isdigit()]
    out=root/f"phase3_v{max(versions,default=0)+1}"; out.mkdir()
    for src,dst in [("feature_schema.json","phase3_feature_schema.json"),("statistics.json","phase3_dataset_statistics.json"),("hashes.json","phase3_data_hashes.json")]: shutil.copyfile(data/src,out/dst)
    interventions=intervention_suite(); interventions["report_version"]=out.name
    (out/"phase3_intervention_tests.json").write_text(json.dumps(interventions,ensure_ascii=False,indent=2),encoding="utf-8")
    integrity=dataset_integrity(data); (out/"phase3_integrity_check.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2),encoding="utf-8")
    stats=json.loads((data/"statistics.json").read_text(encoding="utf-8")); schema=json.loads((data/"feature_schema.json").read_text(encoding="utf-8")); hashes=json.loads((data/"hashes.json").read_text(encoding="utf-8"))
    report=f'''# CTN 第三阶段：数据契约与模拟世界 V2 验收（{out.name}）

## 结果摘要

- 数据版本：`{stats['data_version']}`；固定种子：`{stats['seed']}`。
- 样本规模：train {stats['splits']['train']['samples']:,}、validation {stats['splits']['validation']['samples']:,}、test {stats['splits']['test']['samples']:,}；记录全是成对事实/反事实。
- 测试集同时含常规和预留未见组合样本：OOD 标记样本 {stats['splits']['test']['ood_samples']:,} / {stats['splits']['test']['samples']:,}。
- 单字段直接机制干预：{interventions['passed']}/{interventions['total']} 项符合预期。
- 数据文件 SHA256：见 `phase3_data_hashes.json`。
- 全量完整性检查：配对字段变化、跨 split ID、OOD事实组合与所有 SHA256 检查 `{'通过' if integrity['passed'] else '失败'}`（详见 `phase3_integrity_check.json`）。

## 对验收问题的回答

1. **三个世界均具备显式机制。** Mechanics 的 signed force、摩擦和固定约束计算速度/位移变化；Thermal 的热输入、环境交换、热容和冻结阈值决定温度与物态；Weather 的有效降雨、遮挡、排水及湿度裁剪决定湿度变化。
2. **每字段的作用**已写入 `phase3_feature_schema.json` 和模板文件；数值、布尔、类别有不同明确编码。Mechanics 的 position 是有效状态值但对位移增量不产生影响；其整体平移不变性被作为无效干预测试。
3. **有效因素**：Mechanics X(force/direction/duration)、C(friction/fixed)、S(mass/velocity)；Thermal X(heating/cooling/duration)、C(ambient_temperature)、S(initial_temperature/initial_phase)，freezing_point 只作用 final_phase；Weather X(rainfall/rainfall_duration)、C(roof_cover/drainage)、S(ground_wetness)。
4. **条件性因素**：Mechanics fixed=true 时力不再产生位移；Thermal freezing_point 只在最终温度位于阈值两侧时改变 phase；Weather 初始湿度仅在上下界裁剪条件下改变 wetness_delta。
5. **无效干预**：零降雨时改变 roof_cover；Mechanics 只平移 position 不改变 delta。两者都有自动测试。热学 freezing_point 对温度增量是无效干预，但可改变最终物态。
6. **编码器保留信息**：字段顺序固定，连续值按 train-only 均值/标准差编码，bool 编为 0/1，类别 one-hot 并保留 `__MISSING__`/`__UNK__`；缺失数值用训练均值填补。每字段保留原值到编码位置的 trace。目标归一化按 domain/target 单独拟合，仅 numeric target；final_phase 保留类别标签。
7. **泄漏检查**：pair_id 在 split 内；生成逻辑整对分配后再打乱，pair/world ID 带 split 命名空间。Y_raw/target_state 仅为监督和状态结果，预测输入契约仅包含 X/C/S 编码和 domain。
8. **配对严格性**：同一对只改变 intervention 指定的一个 X/C/S 字段；其余 raw 输入、外部变量与 seed 固定。若干预的是状态变量，state_before 仅该被干预字段相应变化。
9. **未见组合**：测试集一半为训练支持范围内的常规组合，一半使用独立 OOD 条件族：mechanics 高摩擦/力组合、thermal 新冻结点、weather 新降雨/遮挡/排水组合。具体分布和标签在统计 JSON 中。
10. **复现**：`python -m ctn.v2.cli --output data/v2/v2.2 --seed 20261102` 完全确定性；三个 split、模板、因果图、schema、统计文件均写 SHA256。标准化器只在 train rows 上拟合，再可从保存 schema 恢复用于 val/test。

## 简化假设与边界

热学采用常数环境交换系数和“冰热容是液态两倍”的简化规则；不模拟潜热、沸腾、压力、多物质。力学是一维显式更新，不模拟碰撞和非线性摩擦。天气使用线性有效降雨与排水后将湿度裁剪到 `[0,1]`。这些是明确的程序因果机制，不是现实物理的精密模型。

本阶段没有训练任何模型、没有使用测试集调参，也未更改 V1 数据和旧实验报告。测试结果见提交记录/终端，自动化测试文件为 `tests/test_v2_*.py`。
'''
    (out/"phase3_data_report.md").write_text(report,encoding="utf-8")
    return out
