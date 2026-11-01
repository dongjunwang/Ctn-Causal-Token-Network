"""Ground wetness simulator with rain cover and drainage."""
from dataclasses import dataclass


@dataclass
class WeatherWorld:
    ground_wetness: float = 0.0
    timestamp: int = 0
    world_id: str = "W_WEATHER_001"

    def reset(self, ground_wetness=0.0):
        self.ground_wetness, self.timestamp = float(ground_wetness), 0
        return self.observe()

    def observe(self):
        return {"world_id": self.world_id, "timestamp": self.timestamp, "objects": {"ground_001": {"ground_wetness": self.ground_wetness}}, "environment": {}}

    def intervene(self, **changes):
        if "ground_wetness" in changes: self.ground_wetness = float(changes["ground_wetness"])

    def step(self, rainfall=0.0, roof_cover=0.0, drainage=0.0, intervention=False):
        before = self.observe()
        self.ground_wetness = min(1.0, max(0.0, self.ground_wetness + rainfall * (1.0 - roof_cover) - drainage))
        self.timestamp += 1
        return before, self.observe()
