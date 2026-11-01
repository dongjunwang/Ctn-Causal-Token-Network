"""Simplified one-dimensional object motion simulator."""
from dataclasses import dataclass


@dataclass
class MechanicsWorld:
    position: float = 0.0
    velocity: float = 0.0
    fixed: bool = False
    mass: float = 1.0
    timestamp: int = 0
    world_id: str = "W_MECHANICS_001"

    def reset(self, position=0.0, velocity=0.0, fixed=False, mass=1.0):
        self.position, self.velocity, self.fixed, self.mass, self.timestamp = position, velocity, fixed, mass, 0
        return self.observe()

    def observe(self):
        return {"world_id": self.world_id, "timestamp": self.timestamp, "objects": {"object_001": {"position": self.position, "velocity": self.velocity, "fixed": self.fixed, "mass": self.mass}}, "environment": {}}

    def intervene(self, **changes):
        for key in ("position", "velocity", "fixed", "mass"):
            if key in changes: setattr(self, key, changes[key])

    def step(self, force=0.0, friction=0.0, dt=1.0, intervention=False):
        before = self.observe()
        if not self.fixed:
            acceleration = (force - friction * self.velocity) / max(self.mass, 1e-6)
            self.velocity += acceleration * dt
            self.position += self.velocity * dt
        else:
            self.velocity = 0.0
        self.timestamp += 1
        return before, self.observe()
