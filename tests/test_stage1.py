import json
from pathlib import Path
from ctn.models import XEncoder, YEncoder, CEncoder, SEncoder, EventInstance
from ctn.models import EventTemplate, CausalRelation
from ctn.simulator import ThermalWorld, MechanicsWorld, WeatherWorld
from ctn.simulator.generator import generate_dataset


def test_encoders_have_independent_32d_outputs():
    raw = {"force": 2.0}
    for cls in (XEncoder, YEncoder, CEncoder, SEncoder):
        encoded = cls().encode(raw)
        assert encoded.shape == (32,)


def test_mechanics_fixed_object_does_not_move():
    world = MechanicsWorld(); world.reset(fixed=True)
    before, after = world.step(force=5)
    assert before["objects"]["object_001"]["position"] == after["objects"]["object_001"]["position"] == 0


def test_weather_full_cover_blocks_direct_rain():
    world = WeatherWorld(); world.reset()
    _, after = world.step(rainfall=0.8, roof_cover=1.0, drainage=0)
    assert after["objects"]["ground_001"]["ground_wetness"] == 0


def test_thermal_freezing_rule():
    world = ThermalWorld(); world.reset(temperature=5)
    _, after = world.step(cooling_power=1)
    assert after["objects"]["water"]["phase"] == "ice"


def test_seed_reproducibility_and_jsonl(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    generate_dataset(a, 30, 12, 12, 7); generate_dataset(b, 30, 12, 12, 7)
    assert (a / "train.jsonl").read_bytes() == (b / "train.jsonl").read_bytes()
    row = json.loads((a / "train.jsonl").read_text(encoding="utf-8").splitlines()[0])
    EventInstance.model_validate(row["event"])
    assert all(k in row for k in ("X_raw", "Y_raw", "C_raw", "S_raw", "pair_id"))


def test_counterfactual_pairs_share_baseline_and_change_intervention(tmp_path):
    out = tmp_path / "data"; generate_dataset(out, 30, 12, 12, 42)
    rows = [json.loads(line) for line in (out / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    pair = [r for r in rows if r["domain"] == "mechanics"][:2]
    assert pair[0]["pair_id"] == pair[1]["pair_id"]
    assert pair[0]["state_before"] == pair[1]["state_before"]
    assert pair[0]["X_raw"]["force"] != pair[1]["X_raw"]["force"]
    assert {r["intervention"] for r in pair} == {False, True}


def test_test_split_holds_out_mechanism_combinations(tmp_path):
    out = tmp_path / "data"; generate_dataset(out, 300, 90, 90, 42)
    train = [json.loads(line) for line in (out / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [json.loads(line) for line in (out / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    train_combos = {(r["X_raw"]["force"], r["C_raw"]["friction"]) for r in train if r["domain"] == "mechanics"}
    test_combo = {(r["X_raw"]["force"], r["C_raw"]["friction"]) for r in test if r["domain"] == "mechanics" and r["X_raw"]["force"] != 0}
    assert test_combo == {(5.0, 0.8)}
    assert test_combo.isdisjoint(train_combos)


def test_all_generated_records_are_paired(tmp_path):
    out = tmp_path / "data"; report = generate_dataset(out, 60, 30, 30, 8)
    for split in ("train", "validation", "test"):
        rows = [json.loads(line) for line in (out / f"{split}.jsonl").read_text(encoding="utf-8").splitlines()]
        grouped = {}
        for row in rows: grouped.setdefault(row["pair_id"], []).append(row)
        assert all(len(pair) == 2 for pair in grouped.values())
        assert report["splits"][split]["pairs"] == len(rows) // 2
        for a, b in grouped.values():
            assert a["state_before"] == b["state_before"]
            assert a["intervention"] != b["intervention"]


def test_templates_and_causal_graph_cover_planned_types():
    root = Path(__file__).resolve().parents[1]
    templates = sum((json.loads((root / "data" / "templates" / f"{d}.json").read_text(encoding="utf-8")) for d in ("mechanics", "thermal", "weather")), [])
    edges = json.loads((root / "data" / "causal_graph" / "edges.json").read_text(encoding="utf-8"))
    assert len(templates) == 30
    for template in templates: EventTemplate.model_validate(template)
    assert {edge["relation_type"] for edge in edges} == {"CAUSE", "ENABLE", "PREVENT", "CORRELATE", "TEMPORAL"}
    for edge in edges: CausalRelation.model_validate(edge)
