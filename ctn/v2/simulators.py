"""V2 deterministic simulators with explicit, inspectable causal inputs."""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class MechanicsV2:
    def step(self, *, force:float, force_direction:int, duration:float, friction:float, fixed:bool, mass:float, velocity:float, position:float):
        if force_direction not in (-1,1): raise ValueError("force_direction must be -1 or 1")
        if duration < 0 or friction < 0 or mass <= 0: raise ValueError("duration/friction must be nonnegative and mass positive")
        if fixed:
            v1=0.0; dp=0.0
        else:
            acceleration=(force*force_direction-friction*velocity)/mass
            v1=velocity+acceleration*duration
            dp=v1*duration
        return {"velocity_delta":v1-velocity,"displacement_delta":dp,"final_velocity":v1,"final_position":position+dp}


@dataclass
class ThermalV2:
    ambient_exchange:float=0.1
    def step(self, *, heating_power:float, cooling_power:float, duration:float, freezing_point:float, ambient_temperature:float, initial_temperature:float, initial_phase:str):
        if duration < 0: raise ValueError("duration must be nonnegative")
        if initial_phase not in ("ice","liquid"): raise ValueError("initial_phase must be ice or liquid")
        # 简化热容：冰的等效热容设为液态水的 2 倍，体现初始相态对升温速率的影响。
        capacity=2.0 if initial_phase=="ice" else 1.0
        net_rate=heating_power-cooling_power+self.ambient_exchange*(ambient_temperature-initial_temperature)
        delta=net_rate*duration/capacity
        final=initial_temperature+delta
        phase="ice" if final<=freezing_point else "liquid"
        return {"temperature_delta":delta,"final_temperature":final,"final_phase":phase}


@dataclass
class WeatherV2:
    def step(self, *, rainfall:float, rainfall_duration:float, roof_cover:float, drainage:float, ground_wetness:float):
        if rainfall_duration < 0 or rainfall < 0 or drainage < 0: raise ValueError("rainfall, duration and drainage must be nonnegative")
        if not 0<=roof_cover<=1 or not 0<=ground_wetness<=1: raise ValueError("cover and wetness must lie in [0,1]")
        net=rainfall*rainfall_duration*(1-roof_cover)-drainage*rainfall_duration
        final=min(1.0,max(0.0,ground_wetness+net))
        return {"wetness_delta":final-ground_wetness,"final_wetness":final,"unclipped_wetness":ground_wetness+net}
