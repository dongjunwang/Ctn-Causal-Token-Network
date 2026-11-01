from ctn.v2.simulators import MechanicsV2,ThermalV2,WeatherV2

def test_mechanics_each_x_c_s_intervention():
    sim=MechanicsV2(); b=dict(force=2.,force_direction=1,duration=1.,friction=.2,fixed=False,mass=2.,velocity=1.,position=3.)
    y=sim.step(**b)
    assert sim.step(**{**b,"force":3.})["displacement_delta"] != y["displacement_delta"]
    assert sim.step(**{**b,"force_direction":-1})["displacement_delta"] != y["displacement_delta"]
    assert sim.step(**{**b,"duration":2.})["displacement_delta"] != y["displacement_delta"]
    assert sim.step(**{**b,"friction":.8})["displacement_delta"] != y["displacement_delta"]
    assert sim.step(**{**b,"fixed":True})["displacement_delta"] == 0
    assert sim.step(**{**b,"mass":4.})["displacement_delta"] != y["displacement_delta"]
    assert sim.step(**{**b,"velocity":2.})["displacement_delta"] != y["displacement_delta"]
    assert sim.step(**{**b,"position":99.})["displacement_delta"] == y["displacement_delta"]
    shifted=sim.step(**{**b,"position":103.})
    assert shifted["displacement_delta"] == y["displacement_delta"]

def test_thermal_x_c_s_and_freezing_boundary():
    sim=ThermalV2(); b=dict(heating_power=1.,cooling_power=0.,duration=1.,freezing_point=0.,ambient_temperature=0.,initial_temperature=-.5,initial_phase="ice")
    y=sim.step(**b)
    assert sim.step(**{**b,"heating_power":2.})["temperature_delta"] != y["temperature_delta"]
    assert sim.step(**{**b,"cooling_power":.5})["temperature_delta"] != y["temperature_delta"]
    assert sim.step(**{**b,"duration":2.})["temperature_delta"] != y["temperature_delta"]
    ambient={**b,"ambient_temperature":10.}; assert sim.step(**ambient)["temperature_delta"] != y["temperature_delta"]
    # Freeze threshold changes phase but never the energy-derived temperature delta.
    changed=sim.step(**{**b,"freezing_point":2.})
    assert changed["temperature_delta"] == y["temperature_delta"] and changed["final_phase"] != y["final_phase"]
    assert sim.step(**{**b,"initial_temperature":1.})["temperature_delta"] != y["temperature_delta"]
    assert sim.step(**{**b,"initial_phase":"liquid"})["temperature_delta"] != y["temperature_delta"]
    no_cross=sim.step(**{**b,"heating_power":.1})
    assert no_cross["final_phase"] == "ice"

def test_weather_effects_invalid_and_clipping():
    sim=WeatherV2(); b=dict(rainfall=.3,rainfall_duration=1.,roof_cover=0.,drainage=.05,ground_wetness=.4); y=sim.step(**b)
    assert sim.step(**{**b,"rainfall":.5})["wetness_delta"]>y["wetness_delta"]
    assert sim.step(**{**b,"rainfall_duration":2.})["wetness_delta"]>y["wetness_delta"]
    assert sim.step(**{**b,"roof_cover":.5})["wetness_delta"]<y["wetness_delta"]
    assert sim.step(**{**b,"drainage":.2})["wetness_delta"]<y["wetness_delta"]
    assert sim.step(**{**b,"ground_wetness":.8})["final_wetness"] != y["final_wetness"]
    dry={**b,"rainfall":0.}; assert sim.step(**{**dry,"roof_cover":1.}) == sim.step(**{**dry,"roof_cover":0.})
    assert sim.step(**{**b,"ground_wetness":1.})["final_wetness"] == 1.
    assert sim.step(**{**b,"ground_wetness":0.,"rainfall":0.,"drainage":1.})["final_wetness"] == 0.
