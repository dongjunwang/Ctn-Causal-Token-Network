"""Train-fit mixed numeric, boolean and categorical feature encoders."""
from __future__ import annotations
import json, math
from dataclasses import dataclass, asdict
from typing import Any
import numpy as np

MISSING="__MISSING__"; UNKNOWN="__UNK__"

@dataclass
class FieldSpec:
    name:str
    kind:str
    mean:float|None=None
    std:float|None=None
    categories:list[str]|None=None
    missing_fill:float|str|None=None
    offset:int=0
    width:int=1


class StructuredEncoder:
    def __init__(self, version="v2.1"):
        self.version=version; self.fields:list[FieldSpec]=[]; self.fitted=False

    def fit(self, rows:list[dict[str,Any]]):
        keys=sorted({k for row in rows for k in row})
        fields=[]; offset=0
        for key in keys:
            vals=[row.get(key) for row in rows if row.get(key) is not None]
            kinds={"bool" if isinstance(v,bool) else "numeric" if isinstance(v,(int,float)) else "category" for v in vals}
            if not vals: kind="numeric"
            elif len(kinds)!=1: raise TypeError(f"mixed feature types for {key}: {kinds}")
            else: kind=kinds.pop()
            if kind=="numeric":
                numeric=np.asarray([float(v) for v in vals],dtype=np.float64); mean=float(numeric.mean()) if len(numeric) else 0.0; std=float(numeric.std()) if len(numeric) else 1.0
                if not math.isfinite(std) or std<1e-12: std=1.0
                spec=FieldSpec(key,kind,mean,std,None,mean,offset,1); offset+=1
            elif kind=="bool":
                spec=FieldSpec(key,kind,None,None,["false","true"],False,offset,1); offset+=1
            else:
                cats=sorted({str(v) for v in vals})
                # Reserve stable indices for missing and values absent at fit time.
                spec=FieldSpec(key,kind,None,None,cats+[MISSING,UNKNOWN],MISSING,offset,len(cats)+2); offset+=spec.width
            fields.append(spec)
        self.fields=fields; self.dimension=offset; self.fitted=True; return self

    @classmethod
    def from_schema(cls, schema:dict[str,Any]):
        enc=cls(schema["encoder_version"]); enc.fields=[FieldSpec(**f) for f in schema["fields"]]; enc.dimension=schema["dimension"]; enc.fitted=True; return enc

    def transform_one(self, raw:dict[str,Any]):
        if not self.fitted: raise RuntimeError("fit on training data before transform")
        result=np.zeros(self.dimension,dtype=np.float32); trace={}
        for f in self.fields:
            value=raw.get(f.name)
            if f.kind=="numeric":
                raw_value=f.missing_fill if value is None else float(value)
                result[f.offset]=(raw_value-f.mean)/f.std; trace[f.name]={"raw":value,"imputed":raw_value if value is None else None,"encoded_index":f.offset,"encoded_value":float(result[f.offset])}
            elif f.kind=="bool":
                if value is None: value=False
                if not isinstance(value,bool): raise TypeError(f"{f.name} expected bool")
                result[f.offset]=float(value); trace[f.name]={"raw":value,"encoded_index":f.offset,"encoded_value":float(result[f.offset])}
            else:
                token=MISSING if value is None else str(value); cats=f.categories
                if token not in cats: token=UNKNOWN
                idx=cats.index(token); result[f.offset+idx]=1.0; trace[f.name]={"raw":value,"category":token,"encoded_slice":[f.offset,f.offset+f.width],"index":idx}
        return result,trace

    def schema(self):
        return {"encoder_version":self.version,"dimension":self.dimension,"field_order":[f.name for f in self.fields],"fields":[asdict(f) for f in self.fields],"numeric_normalization":"z-score, fitted on train split only","missing_numeric":"training mean imputation","missing_boolean":"false encoded as 0","missing_category":"reserved __MISSING__ one-hot","unseen_category":"reserved __UNK__ one-hot","traceability":"transform_one returns per-field raw value and encoded location/value"}


class TargetNormalizer:
    def __init__(self): self.params={}
    def fit(self, rows):
        grouped={}
        for r in rows:
            for k,v in r["Y_raw"].items():
                if isinstance(v,(int,float)) and not isinstance(v,bool): grouped.setdefault((r["domain"],k),[]).append(float(v))
        for (domain,key),vals in grouped.items():
            a=np.asarray(vals); std=float(a.std()); self.params[f"{domain}.{key}"]={"mean":float(a.mean()),"std":std if std>1e-12 else 1.0,"raw_min":float(a.min()),"raw_max":float(a.max()),"unit":"simulator-defined raw unit"}
        self.params["categorical_targets"]={"thermal.final_phase":{"categories":["ice","liquid"],"encoding":"class label; no numeric normalization"}}
        return self
    def schema(self): return {"fit_scope":"training split only","targets":self.params}
    def transform(self,domain,key,value):
        p=self.params[f"{domain}.{key}"]; return (float(value)-p["mean"])/p["std"]
    def inverse_transform(self,domain,key,value):
        p=self.params[f"{domain}.{key}"]; return float(value)*p["std"]+p["mean"]
