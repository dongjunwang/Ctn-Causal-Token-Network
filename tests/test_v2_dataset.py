import hashlib,json
from ctn.v2.dataset import generate_v2
from ctn.v2.contracts import CausalSample
from ctn.v2.encoders import StructuredEncoder

def test_v2_dataset_pairs_splits_reproduce_and_hash(tmp_path):
    a=tmp_path/"a"; b=tmp_path/"b"
    generate_v2(a,120,30,30,77); generate_v2(b,120,30,30,77)
    for f in ("train.jsonl","validation.jsonl","test.jsonl"):
        assert (a/f).read_bytes()==(b/f).read_bytes()
    rows={s:[json.loads(x) for x in (a/f"{s}.jsonl").read_text(encoding="utf-8").splitlines()] for s in ("train","validation","test")}
    pairsets={s:{r["pair_id"] for r in rs} for s,rs in rows.items()}
    assert not (pairsets["train"]&pairsets["validation"] or pairsets["train"]&pairsets["test"] or pairsets["validation"]&pairsets["test"])
    for split,rs in rows.items():
        grouped={}
        for r in rs: grouped.setdefault(r["pair_id"],[]).append(r)
        assert all(len(pair)==2 for pair in grouped.values())
        for pair in grouped.values():
            pair.sort(key=lambda r:r["intervention"]["is_intervention"])
            CausalSample.model_validate(pair[0])
            intervention_feature=pair[1]["intervention"]["feature"]
            if intervention_feature in pair[0]["S_raw"]:
                changed_state=[k for k in pair[0]["S_raw"] if pair[0]["S_raw"][k]!=pair[1]["S_raw"][k]]
                assert changed_state==[intervention_feature]
            else: assert pair[0]["state_before"]==pair[1]["state_before"]
            changes=[k for k in set(pair[0]["X_raw"])|set(pair[0]["C_raw"])|set(pair[0]["S_raw"]) if pair[0]["X_raw"].get(k)!=pair[1]["X_raw"].get(k) or pair[0]["C_raw"].get(k)!=pair[1]["C_raw"].get(k) or pair[0]["S_raw"].get(k)!=pair[1]["S_raw"].get(k)]
            assert len(changes)==1
    hashes=json.loads((a/"hashes.json").read_text(encoding="utf-8"))
    for filename,digest in hashes["files"].items(): assert hashlib.sha256((a/filename).read_bytes()).hexdigest()==digest

def test_train_only_schema_and_model_inputs_exclude_labels(tmp_path):
    a=tmp_path/"d"; generate_v2(a,60,30,30,8)
    schema=json.loads((a/"feature_schema.json").read_text(encoding="utf-8"))
    assert schema["fit_split"]=="train"
    assert schema["prediction_inputs"]==["X_encoded","C_encoded","S_encoded","domain"]
    assert "Y_raw" in schema["supervision_only"] and "target_state" in schema["supervision_only"]
    rows=[json.loads(x) for x in (a/"train.jsonl").read_text(encoding="utf-8").splitlines()]
    enc=StructuredEncoder().fit([r["X_raw"] for r in rows if r["domain"]=="mechanics"])
    assert enc.transform_one(rows[0]["X_raw"])[0].ndim==1
    sample=CausalSample.model_validate(rows[0]); model_inputs=sample.model_inputs()
    assert not ({"Y_raw","target_state"}&set(model_inputs))
