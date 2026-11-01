import inspect

import torch

from models.v2.models import CTN, MLPFull, TransformerFull, XY


def _inputs(batch=5):
    return (
        torch.randn(batch, 10),
        torch.randn(batch, 8),
        torch.randn(batch, 6),
        torch.eye(3)[torch.arange(batch) % 3],
    )


def test_four_models_accept_only_xcs_and_domain_and_emit_all_heads():
    dims = (10, 8, 6)
    models = [XY(*dims), MLPFull(*dims, h=88), TransformerFull(*dims, h=32), CTN(*dims, h=29)]
    assert all(list(inspect.signature(model.forward).parameters) == ["x", "c", "s", "d"] for model in models)
    for model in models:
        outputs = model(*_inputs())
        assert set(outputs) == {"y_num", "y_phase", "s_num", "s_phase"}
        assert outputs["y_num"].shape == (5, 6)
        assert outputs["y_phase"].shape == (5, 2)
        assert outputs["s_num"].shape == (5, 4)
        assert outputs["s_phase"].shape == (5, 2)


def test_full_model_parameter_counts_are_within_ten_percent():
    # V2.2 maximum padded widths from the saved feature schema.
    dims = (3, 2, 5)
    models = [MLPFull(*dims, h=88), TransformerFull(*dims, h=32), CTN(*dims, h=29)]
    counts = [sum(p.numel() for p in model.parameters()) for model in models]
    assert max(counts) / min(counts) - 1 < 0.10


def test_ctn_exposes_separate_encoders_relation_gates_and_transition_module():
    model = CTN(10, 8, 6, h=29)
    assert model.xe is not model.ce and model.ce is not model.se
    assert all(hasattr(model, gate) for gate in ("g_xc", "g_xs", "g_cs", "g_all"))
    assert model.transition is not None
    outputs = model(*_inputs())
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert {"gate_xc", "gate_xs", "gate_cs", "gate_all"} <= model.activations.keys()


def test_ctn_ablation_switches_remove_requested_components():
    model = CTN(10, 8, 6, h=29, ablation={"no_c", "no_s", "no_transition"})
    assert model.ce is None and model.se is None and model.transition is None
    outputs = model(*_inputs())
    assert all(torch.isfinite(value).all() for value in outputs.values())
