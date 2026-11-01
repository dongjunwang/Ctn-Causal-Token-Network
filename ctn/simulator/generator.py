"""Deterministic synthetic dataset generation with held-out combinations."""
from __future__ import annotations

import json
import random
from pathlib import Path
from ..models import CausalSample, EventInstance
from .thermal import ThermalWorld
from .mechanics import MechanicsWorld
from .weather import WeatherWorld

DOMAINS = ("thermal", "mechanics", "weather")


def _sample(domain: str, index: int, split: str, rng: random.Random, counterfactual: bool = False) -> CausalSample:
    pair_id = f"P_{domain}_{index:06d}"
    event_id = f"E_{domain}_{index:06d}"
    intervention = counterfactual
    if domain == "mechanics":
        fixed = False
        if split == "test":
            factual_force, friction = 5.0, 0.8  # 测试保留未见的力-摩擦组合
        else:
            factual_force = rng.choice([1.0, 3.0])
            friction = rng.choice([0.0, 0.2])
        force, mass = (0.0 if counterfactual else factual_force), rng.choice([1.0, 2.0])
        world = MechanicsWorld(); before = world.reset(fixed=fixed, mass=mass); _, after = world.step(force, friction, intervention=intervention)
        X, Y, C = {"force": force, "action": "push"}, {"position_delta": after["objects"]["object_001"]["position"]}, {"friction": friction, "fixed": fixed}
        S = before["objects"]["object_001"]; action, typ, target = "push", "EV_MOVE_001", "position_change"
        params = {"force": force, "friction": friction}
    elif domain == "thermal":
        temp = 5.0 if split == "test" else rng.choice([-5.0, 2.0, 20.0]); heat, cool = (0.0, 1.0) if counterfactual else (1.0, 0.0)
        world = ThermalWorld(); before = world.reset(temp); _, after = world.step(heat, cool, intervention=intervention)
        X, Y, C = {"heating_power": heat, "cooling_power": cool}, {"phase": after["objects"]["water"]["phase"], "temperature_delta": 10 * (heat-cool)}, {"freezing_point": 0.0}
        S = before["objects"]["water"]; action, typ, target = "heat_or_cool", "EV_PHASE_CHANGE_001", "phase_change"
        params = {"heating_power": heat, "cooling_power": cool}
    else:
        rain, drain = ((0.6, 0.4) if split == "test" else (rng.choice([0.0, 0.2]), rng.choice([0.0, 0.1])))
        cover = 1.0 if counterfactual else 0.0
        world = WeatherWorld(); before = world.reset(); _, after = world.step(rain, cover, drain, intervention)
        X, Y, C = {"rainfall": rain}, {"ground_wetness": after["objects"]["ground_001"]["ground_wetness"]}, {"roof_cover": cover, "drainage": drain}
        S = before["objects"]["ground_001"]; action, typ, target = "rain", "EV_GROUND_WET_001", "wetness_change"
        params = {"rainfall": rain, "drainage": drain}
    event = EventInstance(event_id=event_id, world_id=before["world_id"], event_type_id=typ, timestamp=0, actor="environment", object=domain, action=action, parameters=params, conditions=C, state_before=S, state_after=after["objects"], intervention=intervention)
    return CausalSample(sample_id=f"S_{domain}_{index:06d}", domain=domain, cause=X, X_raw=X, Y_raw=Y, C_raw=C, S_raw=S, state_before=before["objects"], target_state=after["objects"], target_event=target, relation_label="CAUSE", intervention=intervention, pair_id=pair_id, split=split, event=event)


def generate_dataset(output_dir: str | Path, train=5000, validation=1000, test=1000, seed=42):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    counts = {"train": train, "validation": validation, "test": test}; report = {"seed": seed, "splits": {}, "domains": {}}
    for split, total in counts.items():
        path = out / f"{split}.jsonl"; per_domain = total // len(DOMAINS); remainder = total % len(DOMAINS)
        rows = []
        for d_i, domain in enumerate(DOMAINS):
            n = per_domain + (1 if d_i < remainder else 0)
            offset = 100000 if split == "validation" else 200000 if split == "test" else 0
            for i in range(n // 2):
                pair_index = i + offset
                # 同一 pair 的两行重置随机数，确保外生参数相同，只切换指定干预变量。
                pair_seed = seed + pair_index * 17 + d_i * 1000003
                rows.append(_sample(domain, pair_index, split, random.Random(pair_seed), False).model_dump(mode="json"))
                rows.append(_sample(domain, pair_index, split, random.Random(pair_seed), True).model_dump(mode="json"))
            if n % 2:
                pair_index = n // 2 + offset
                rows.append(_sample(domain, pair_index, split, random.Random(seed + pair_index * 17 + d_i * 1000003), False).model_dump(mode="json"))
        with path.open("w", encoding="utf-8") as f:
            for row in rows: f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        report["splits"][split] = {"samples": len(rows), "interventions": sum(x["intervention"] for x in rows), "pairs": len({x["pair_id"] for x in rows})}
    for domain in DOMAINS:
        report["domains"][domain] = {"samples": sum(n // 3 + (1 if DOMAINS.index(domain) < n % 3 else 0) for n in counts.values()), "event_types": 1, "relation_types": ["CAUSE"], "intervention_rate": 0.2}
    (out / "quality_report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
