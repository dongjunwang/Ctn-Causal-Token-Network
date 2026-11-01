"""Simplified water heating/freezing simulator."""
from dataclasses import dataclass


@dataclass
class ThermalWorld:
    temperature: float = 20.0
    phase: str = "liquid"
    timestamp: int = 0
    world_id: str = "W_THERMAL_001"

    def reset(self, temperature: float = 20.0, phase: str | None = None):
        self.temperature, self.phase, self.timestamp = temperature, phase or ("ice" if temperature <= 0 else "liquid"), 0
        return self.observe()

    def observe(self):
        return {"world_id": self.world_id, "timestamp": self.timestamp, "objects": {"water": {"temperature": self.temperature, "phase": self.phase}}, "environment": {}}

    def intervene(self, **changes):
        if "temperature" in changes: self.temperature = float(changes["temperature"])
        if "phase" in changes: self.phase = str(changes["phase"])

    def step(self, heating_power=0.0, cooling_power=0.0, dt=1.0, intervention=False):
        before = self.observe()
        self.temperature += 10.0 * (heating_power - cooling_power) * dt
        if self.temperature <= 0: self.phase = "ice"
        elif self.temperature > 0: self.phase = "liquid"
        self.timestamp += 1
        return before, self.observe()
