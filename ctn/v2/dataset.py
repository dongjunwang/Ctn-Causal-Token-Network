"""Paired V2 dataset generation, train-only feature fitting, and manifests."""
from __future__ import annotations
import hashlib,json,random
from pathlib import Path
from collections import defaultdict
from .simulators import MechanicsV2,ThermalV2,WeatherV2
from .encoders import StructuredEncoder,TargetNormalizer
from .contracts import CausalSample,EventTemplate

VERSION="v2.2"
DOMAINS=("mechanics","thermal","weather")

def _mechanics_case(rng,ood=False):
    if ood: return dict(force=3.0,force_direction=1,duration=1.5,friction=.9,fixed=False,mass=4.0,velocity=1.0,position=2.0)
    return dict(force=rng.choice([.5,1.,2.,3.]),force_direction=rng.choice([-1,1]),duration=rng.choice([.5,1.]),friction=rng.choice([0.,.2,.5]),fixed=rng.choice([False,True]),mass=rng.choice([1.,2.,4.]),velocity=rng.choice([-1.,0.,1.]),position=rng.choice([-2.,0.,2.]))

def _thermal_case(rng,ood=False):
    if ood: freezing=5.0; temp=rng.choice([-3.,4.,8.]); ambient=10.0
    else: freezing=rng.choice([-5.,0.]); temp=freezing+rng.choice([-6.,-2.,2.,6.]); ambient=rng.choice([-5.,5.])
    phase="ice" if temp<=freezing else "liquid"
    return dict(heating_power=rng.choice([0.,.5,1.]),cooling_power=rng.choice([0.,.25,.75]),duration=rng.choice([.5,1.,2.]),freezing_point=freezing,ambient_temperature=ambient,initial_temperature=temp,initial_phase=phase)

def _weather_case(rng,ood=False):
    if ood: return dict(rainfall=.8,rainfall_duration=2.,roof_cover=.25,drainage=.3,ground_wetness=rng.choice([.05,.5,.95]))
    return dict(rainfall=rng.choice([0.,.1,.3,.5]),rainfall_duration=rng.choice([.5,1.,2.]),roof_cover=rng.choice([0.,.5,1.]),drainage=rng.choice([0.,.05,.2]),ground_wetness=rng.choice([0.,.2,.5,.8,1.]))

def _intervention(domain, base, rng, ood=False):
    changed=dict(base)
    if domain=="mechanics":
        feature=rng.choice(["friction","force","fixed","position"])
        if feature=="friction": changed[feature]=.9 if base[feature]!=.9 else 0.0
        elif feature=="force": changed[feature]=0.0 if base[feature]!=0.0 else 3.0
        elif feature=="fixed": changed[feature]=not base[feature]
        else: changed[feature]=base[feature]+10.0
    elif domain=="thermal":
        feature=rng.choice(["initial_temperature","freezing_point","heating_power","ambient_temperature"])
        if feature=="initial_temperature": changed[feature]=base[feature]+4.0
        elif feature=="freezing_point": changed[feature]=base[feature]+3.0
        elif feature=="heating_power": changed[feature]=base[feature]+.5
        else: changed[feature]=base[feature]+10.0
        # initial_phase is held fixed: exactly one feature changes in a pair.
    else:
        feature=rng.choice(["rainfall","roof_cover","drainage","ground_wetness"])
        if feature=="rainfall": changed[feature]=.8 if base[feature]!=.8 else 0.0
        elif feature=="roof_cover": changed[feature]=1.0 if base[feature]!=1.0 else 0.0
        elif feature=="drainage": changed[feature]=.4 if base[feature]!=.4 else 0.0
        else: changed[feature]=1.0 if base[feature]<0.5 else 0.0
    if domain=="weather" and feature=="roof_cover" and base["rainfall"]==0: kind="NO_EFFECT"
    elif domain=="mechanics" and feature=="position": kind="NO_EFFECT"
    elif domain=="thermal" and feature=="freezing_point": kind="CONDITIONALLY_CAUSAL"
    elif domain=="weather" and feature=="ground_wetness": kind="CONDITIONALLY_CAUSAL"
    else: kind="CAUSE"
    return feature,changed,kind

def _run(domain, f):
    if domain=="mechanics":
        x={k:f[k] for k in ("force","force_direction","duration")}; c={k:f[k] for k in ("friction","fixed")}; s={k:f[k] for k in ("mass","velocity","position")}
        y=MechanicsV2().step(**f)
    elif domain=="thermal":
        x={k:f[k] for k in ("heating_power","cooling_power","duration")}; c={k:f[k] for k in ("freezing_point","ambient_temperature")}; s={k:f[k] for k in ("initial_temperature","initial_phase")}
        y=ThermalV2().step(**f)
    else:
        x={k:f[k] for k in ("rainfall","rainfall_duration")}; c={k:f[k] for k in ("roof_cover","drainage")}; s={"ground_wetness":f["ground_wetness"]}
        y=WeatherV2().step(**f)
    return x,c,s,y

def _sample(domain,pair_index,split,ood,seed,rng):
    base=(_mechanics_case if domain=="mechanics" else _thermal_case if domain=="thermal" else _weather_case)(rng,ood)
    feat,changed,kind=_intervention(domain,base,rng,ood)
    pair_id=f"{VERSION}_{split}_{domain}_{pair_index:06d}"; world_id=f"W_{VERSION}_{split}_{domain}_{pair_index:06d}"
    _,_,_,y_base=_run(domain,base); _,_,_,y_changed=_run(domain,changed)
    if y_base==y_changed: kind="NO_EFFECT"
    elif (domain=="thermal" and feat=="freezing_point") or (domain=="weather" and feat=="ground_wetness") or (domain=="weather" and feat=="roof_cover" and base["rainfall"]==0) or (domain=="mechanics" and feat in ("force","force_direction","duration","friction","fixed")):
        kind="CONDITIONALLY_CAUSAL"
    else: kind="CAUSE"
    rows=[]
    for variant,features in ((False,base),(True,changed)):
        x,c,s,y=_run(domain,features)
        event_id=f"E_{pair_id}_{int(variant)}"; event_type={"mechanics":"EV_MOTION_V2","thermal":"EV_THERMAL_V2","weather":"EV_WETNESS_V2"}[domain]
        state_before={"world_id":world_id,"timestamp":0,"objects":{"system":s},"environment":{k:v for k,v in features.items() if k not in set(x)|set(c)|set(s)}}
        target_state={"objects":{"system":{**s,**y}},"environment":state_before["environment"]}
        intervention={"is_intervention":variant,"feature":feat if variant else None,"value_before":base[feat] if variant else None,"value_after":changed[feat] if variant else None,"pair_control":"counterfactual" if variant else "factual"}
        event={"event_id":event_id,"world_id":world_id,"event_type_id":event_type,"timestamp":0,"actor":"simulator","object":domain,"action":event_type,"parameters":x,"conditions":c,"state_before":state_before["objects"],"state_after":target_state["objects"],"intervention":variant}
        row={"sample_id":f"S_{pair_id}_{int(variant)}","pair_id":pair_id,"world_id":world_id,"domain":domain,"event":event,"X_raw":x,"C_raw":c,"S_raw":s,"Y_raw":y,"intervention":intervention,"relation_label":kind,"state_before":state_before,"target_state":target_state,"evidence_type":"simulator","split":split,"ood_condition":bool(ood)}
        rows.append(CausalSample.model_validate(row).model_dump(mode="json"))
    return rows

def _features(rows,group): return [r[f"{group}_raw"] for r in rows]

def generate_v2(output_dir="data/v2/v2.2",train=50000,validation=10000,test=10000,seed=20261102):
    out=Path(output_dir); out.mkdir(parents=True,exist_ok=True)
    template_dir=out/"templates"; graph_dir=out/"causal_graph"; template_dir.mkdir(exist_ok=True); graph_dir.mkdir(exist_ok=True)
    template_rows={
      "mechanics":[{"event_type_id":"EV_MOTION_V2","name":"一维外力运动","domain":"mechanics","actor":"external_force","action":"accelerate","object":"body","cause_features":["force","force_direction","duration","friction","fixed","mass","velocity"],"effect_features":["velocity_delta","displacement_delta"]}],
      "thermal":[{"event_type_id":"EV_THERMAL_V2","name":"水的简化热交换与相变","domain":"thermal","actor":"heater_environment","action":"heat_exchange","object":"water","cause_features":["heating_power","cooling_power","duration","freezing_point","ambient_temperature","initial_temperature","initial_phase"],"effect_features":["temperature_delta","final_temperature","final_phase"]}],
      "weather":[{"event_type_id":"EV_WETNESS_V2","name":"降雨与地面湿度变化","domain":"weather","actor":"weather","action":"precipitate_and_drain","object":"ground","cause_features":["rainfall","rainfall_duration","roof_cover","drainage","ground_wetness"],"effect_features":["wetness_delta","final_wetness"]}],
    }
    templates=[EventTemplate.model_validate(x).model_dump(mode="json") for rows in template_rows.values() for x in rows]
    for domain,items in template_rows.items(): (template_dir/f"{domain}.json").write_text(json.dumps(items,ensure_ascii=False,indent=2),encoding="utf-8")
    nodes=[{"id":x["event_type_id"],"domain":x["domain"],"name":x["name"]} for x in templates]
    edges=[
      {"relation_id":"R_MOTION_FORCE","source":"EV_MOTION_V2","target":"EV_MOTION_V2","relation_type":"CAUSE","condition":{"fixed":False},"mechanism":"signed external force changes velocity and displacement","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
      {"relation_id":"R_MOTION_FRICTION","source":"EV_MOTION_V2","target":"EV_MOTION_V2","relation_type":"CAUSE","condition":{"velocity_nonzero":True},"mechanism":"friction opposes velocity","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
      {"relation_id":"R_MOTION_FIXED","source":"EV_MOTION_V2","target":"EV_MOTION_V2","relation_type":"PREVENT","condition":{"fixed":True},"mechanism":"fixed constraint prevents displacement","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
      {"relation_id":"R_THERMAL_ENERGY","source":"EV_THERMAL_V2","target":"EV_THERMAL_V2","relation_type":"CAUSE","condition":{},"mechanism":"heating, cooling, ambient exchange and heat capacity determine temperature change","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
      {"relation_id":"R_THERMAL_THRESHOLD","source":"EV_THERMAL_V2","target":"EV_THERMAL_V2","relation_type":"ENABLE","condition":{"final_temperature_crosses_freezing_point":True},"mechanism":"freezing threshold determines phase label without changing energy balance","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
      {"relation_id":"R_WEATHER_RAIN","source":"EV_WETNESS_V2","target":"EV_WETNESS_V2","relation_type":"CAUSE","condition":{"roof_cover_lt_1":True},"mechanism":"exposed rainfall adds ground water","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
      {"relation_id":"R_WEATHER_DRAIN","source":"EV_WETNESS_V2","target":"EV_WETNESS_V2","relation_type":"CAUSE","condition":{},"mechanism":"drainage subtracts water before clipping","confidence":1.0,"evidence_type":"simulator","intervention_tested":True},
    ]
    (graph_dir/"nodes.json").write_text(json.dumps(nodes,ensure_ascii=False,indent=2),encoding="utf-8"); (graph_dir/"edges.json").write_text(json.dumps(edges,ensure_ascii=False,indent=2),encoding="utf-8")
    if any(n%2 for n in (train,validation,test)): raise ValueError("split record counts must be even for complete pairs")
    splits={"train":train,"validation":validation,"test":test}; records={}; rng=random.Random(seed)
    for split,n in splits.items():
        pairs=n//2; per=pairs//3; rem=pairs%3; allrows=[]
        for di,domain in enumerate(DOMAINS):
            count=per+(di<rem); drows=[]
            for i in range(count):
                # Test set includes both familiar combinations and held-out combinations.
                ood=(split=="test" and i%2==1)
                drows.extend(_sample(domain,i,split,ood,seed,rng))
            allrows.extend(drows)
        # Shuffle records only after pair construction; pair identity never crosses a split.
        rng.shuffle(allrows); records[split]=allrows
        with (out/f"{split}.jsonl").open("w",encoding="utf-8",newline="\n") as f:
            for row in allrows: f.write(json.dumps(row,ensure_ascii=False,separators=(",",":"))+"\n")
    trainrows=records["train"]
    encoders={}
    for domain in DOMAINS:
        dr=[r for r in trainrows if r["domain"]==domain]
        encoders[domain]={g:StructuredEncoder().fit(_features(dr,g)) for g in ("X","C","S")}
    schema={"data_version":VERSION,"fit_split":"train","domains":{d:{g:e.schema() for g,e in encoders[d].items()} for d in DOMAINS},"targets":TargetNormalizer().fit(trainrows).schema(),"prediction_inputs":["X_encoded","C_encoded","S_encoded","domain"],"supervision_only":["Y_raw","target_state"],"raw_fields_retained":True}
    (out/"feature_schema.json").write_text(json.dumps(schema,ensure_ascii=False,indent=2),encoding="utf-8")
    stats={"data_version":VERSION,"seed":seed,"splits":{},"domains":{},"field_distributions":{}}
    for split,rows in records.items():
        stats["splits"][split]={"samples":len(rows),"pairs":len({r['pair_id'] for r in rows}),"interventions":sum(r['intervention']['is_intervention'] for r in rows),"ood_samples":sum(r['ood_condition'] for r in rows),"domains":{d:sum(r['domain']==d for r in rows) for d in DOMAINS},"intervention_features":dict(__import__('collections').Counter(r['intervention']['feature'] for r in rows if r['intervention']['is_intervention']))}
    for domain in DOMAINS:
        stats["domains"][domain]={}
        for split,rows in records.items():
            rs=[r for r in rows if r["domain"]==domain]
            stats["domains"][domain][split]={"samples":len(rs),"pairs":len({r['pair_id'] for r in rs}),"interventions":sum(r['intervention']['is_intervention'] for r in rs),"ood_samples":sum(r['ood_condition'] for r in rs),"relation_labels":dict(__import__('collections').Counter(r['relation_label'] for r in rs))}
        for split,split_rows in records.items():
          for group in ("X","C","S","Y"):
            field_names=sorted({k for r in split_rows if r['domain']==domain for k in r[f"{group}_raw"]})
            for field in field_names:
                vals=[r[f"{group}_raw"].get(field) for r in split_rows if r['domain']==domain]
                nums=[float(v) for v in vals if isinstance(v,(int,float)) and not isinstance(v,bool)]
                stats["field_distributions"][f"{split}.{domain}.{group}.{field}"]={"kind":"numeric" if nums else "category/boolean","count":len(vals),"unique":len({str(v) for v in vals}),"min":min(nums) if nums else None,"max":max(nums) if nums else None,"mean":sum(nums)/len(nums) if nums else None}
    (out/"statistics.json").write_text(json.dumps(stats,ensure_ascii=False,indent=2),encoding="utf-8")
    hashes={"data_version":VERSION,"seed":seed,"files":{}}
    for name in ("train.jsonl","validation.jsonl","test.jsonl","feature_schema.json","statistics.json","templates/mechanics.json","templates/thermal.json","templates/weather.json","causal_graph/nodes.json","causal_graph/edges.json"):
        hashes["files"][name]=hashlib.sha256((out/name).read_bytes()).hexdigest()
    (out/"hashes.json").write_text(json.dumps(hashes,ensure_ascii=False,indent=2),encoding="utf-8")
    return stats,hashes,schema
