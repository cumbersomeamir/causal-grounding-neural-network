from pydantic import BaseModel
from typing import Dict, List, Optional, Any

class PredictionRequest(BaseModel):
    data: Dict[str, float]

class InterventionRequest(BaseModel):
    intervention: Dict[str, float]
    n_samples: int = 100

class CounterfactualRequest(BaseModel):
    observation: Dict[str, float]
    intervention: Dict[str, float]

class GraphResponse(BaseModel):
    nodes: List[str]
    edges: List[List[str]]

class SampleRequest(BaseModel):
    n_samples: int = 100

class ActiveLearningRequest(BaseModel):
    current_data: List[Dict[str, float]]
    n_ensemble: int = 5

class AdaptRequest(BaseModel):
    new_data: List[Dict[str, float]]
    steps: int = 5
