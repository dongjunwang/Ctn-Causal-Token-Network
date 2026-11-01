"""Validated V2 sample contracts. Labels/state are separated from model inputs."""
from typing import Any,Literal
from pydantic import BaseModel,ConfigDict,Field

class StrictModel(BaseModel):
    model_config=ConfigDict(extra="forbid")

class EventTemplate(StrictModel):
    event_type_id:str; name:str; domain:str; actor:str; action:str; object:str; cause_features:list[str]; effect_features:list[str]

class EventInstance(StrictModel):
    event_id:str; world_id:str; event_type_id:str; timestamp:int; actor:str; object:str; action:str; parameters:dict[str,Any]; conditions:dict[str,Any]; state_before:dict[str,Any]; state_after:dict[str,Any]; intervention:bool

class WorldState(StrictModel):
    world_id:str; timestamp:int; objects:dict[str,Any]; environment:dict[str,Any]

class CausalRelation(StrictModel):
    relation_id:str; source:str; target:str; relation_type:Literal["CAUSE","ENABLE","PREVENT","CORRELATE","TEMPORAL"]; condition:dict[str,Any]; mechanism:str; confidence:float=Field(ge=0,le=1); evidence_type:str; intervention_tested:bool

class CausalSample(StrictModel):
    sample_id:str; pair_id:str; world_id:str; domain:Literal["mechanics","thermal","weather"]; event:EventInstance; X_raw:dict[str,Any]; C_raw:dict[str,Any]; S_raw:dict[str,Any]; Y_raw:dict[str,Any]; intervention:dict[str,Any]; relation_label:str; state_before:WorldState; target_state:dict[str,Any]; evidence_type:str; split:Literal["train","validation","test"]; ood_condition:bool

    def model_inputs(self):
        return {"X_raw":self.X_raw,"C_raw":self.C_raw,"S_raw":self.S_raw,"domain":self.domain}
