import numpy as np
import pytest
from ctn.v2.encoders import StructuredEncoder,TargetNormalizer

def test_mixed_encoder_fixed_width_categories_and_trace():
    enc=StructuredEncoder().fit([{"amount":1.,"flag":True,"kind":"ice"},{"amount":3.,"flag":False,"kind":"liquid"}])
    assert enc.dimension==6
    vector,trace=enc.transform_one({"amount":2.,"flag":True,"kind":"ice"})
    assert vector.shape==(6,); assert vector[trace["flag"]["encoded_index"]]==1
    assert vector[trace["kind"]["encoded_slice"][0]+trace["kind"]["index"]]==1
    assert trace["amount"]["raw"]==2.

def test_missing_and_unknown_are_explicit():
    enc=StructuredEncoder().fit([{"n":1.,"cat":"a"},{"n":3.,"cat":"b"}])
    arr,trace=enc.transform_one({"cat":"new"})
    assert trace["n"]["imputed"]==2.
    assert trace["cat"]["category"]=="__UNK__"
    assert np.isfinite(arr).all()

def test_encoder_requires_fit():
    with pytest.raises(RuntimeError): StructuredEncoder().transform_one({"x":1})

def test_encoder_roundtrips_saved_schema():
    enc=StructuredEncoder().fit([{"x":1.},{"x":3.}])
    restored=StructuredEncoder.from_schema(enc.schema())
    assert np.array_equal(enc.transform_one({"x":2.})[0],restored.transform_one({"x":2.})[0])

def test_target_normalizer_is_per_domain_target():
    rows=[{"domain":"mechanics","Y_raw":{"dy":1.}},{"domain":"mechanics","Y_raw":{"dy":3.}},{"domain":"weather","Y_raw":{"wet":.2}}]
    n=TargetNormalizer().fit(rows).schema()["targets"]
    assert n["mechanics.dy"]["mean"]==2.
    assert n["weather.wet"]["mean"]==.2
    normalizer=TargetNormalizer().fit(rows)
    assert normalizer.inverse_transform("mechanics","dy",normalizer.transform("mechanics","dy",3.))==3.
