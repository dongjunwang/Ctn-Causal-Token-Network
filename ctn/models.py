"""Core CTN records and deterministic 32-dimensional feature encoders."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EventTemplate(Record):
    event_type_id: str
    name: str
    domain: str
    actor: str
    action: str
    object: str
    cause_features: list[str]
    effect_features: list[str]


class EventInstance(Record):
    event_id: str
    world_id: str
    event_type_id: str
    timestamp: int
    actor: str
    object: str
    action: str
    parameters: dict[str, Any]
    conditions: dict[str, Any]
    state_before: dict[str, Any]
    state_after: dict[str, Any]
    intervention: bool = False


class WorldState(Record):
    world_id: str
    timestamp: int
    objects: dict[str, Any]
    environment: dict[str, Any]


class CausalRelation(Record):
    relation_id: str
    source: str
    target: str
    relation_type: Literal["CAUSE", "ENABLE", "PREVENT", "CORRELATE", "TEMPORAL"]
    condition: dict[str, Any] = Field(default_factory=dict)
    mechanism: str
    confidence: float = Field(ge=0, le=1)
    evidence_type: str
    intervention_tested: bool


class CausalSample(Record):
    sample_id: str
    domain: str
    cause: dict[str, Any]
    X_raw: dict[str, Any]
    Y_raw: dict[str, Any]
    C_raw: dict[str, Any]
    S_raw: dict[str, Any]
    state_before: dict[str, Any]
    target_state: dict[str, Any]
    target_event: str
    relation_label: str
    intervention: bool
    pair_id: str
    split: str
    event: EventInstance


class FeatureEncoder:
    """Stable feature hashing to 32 dims; raw values remain the source of truth."""
    def __init__(self, dimensions: int = 32):
        self.dimensions = dimensions

    def encode(self, raw: dict[str, Any]) -> np.ndarray:
        out = np.zeros(self.dimensions, dtype=np.float32)
        for key, value in sorted(raw.items()):
            payload = json.dumps([key, value], sort_keys=True, ensure_ascii=False, default=str)
            digest = hashlib.blake2b(payload.encode(), digest_size=16).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            magnitude = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 1.0
            out[index] += sign * magnitude
        norm = np.linalg.norm(out)
        return out / norm if norm else out


class XEncoder(FeatureEncoder): pass
class YEncoder(FeatureEncoder): pass
class CEncoder(FeatureEncoder): pass
class SEncoder(FeatureEncoder): pass
