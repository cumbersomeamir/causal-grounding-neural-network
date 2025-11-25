from fastapi import FastAPI, HTTPException, BackgroundTasks
import uvicorn
import torch
import pickle
import pandas as pd
import networkx as nx
from typing import List
import os

from .api import (
    PredictionRequest, InterventionRequest, CounterfactualRequest, 
    GraphResponse, SampleRequest, ActiveLearningRequest, AdaptRequest
)
from ..causal_graph.scm import SCM
from ..causal_graph.structure_learning import load_graph
from ..causal_graph.discovery.notears import DeepCausalDiscovery
from ..active_learning.agent import ActiveCausalAgent
from ..meta_learning.maml import MetaSCM
from ..utils.logger import get_logger

app = FastAPI(title="Causal AGI API", description="Advanced Causal Reasoning & Discovery Service")
logger = get_logger(__name__)

# Global state
scm_model = None
meta_scm = None
preprocessor = None
active_agent = None

@app.on_event("startup")
def load_models():
    global scm_model, meta_scm, preprocessor, active_agent
    try:
        graph_path = "artifacts/graph.pkl"
        model_path = "artifacts/scm_model.pth"
        preprocessor_path = "artifacts/preprocessor.pkl"
        
        if os.path.exists(graph_path) and os.path.exists(model_path):
            graph = load_graph(graph_path)
            scm_model = SCM(graph)
            scm_model.load_state_dict(torch.load(model_path))
            scm_model.eval()
            
            # Initialize Meta SCM
            meta_scm = MetaSCM(scm_model)
            
            logger.info("SCM and Meta-Learner loaded successfully.")
        
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
                
        # Initialize Active Agent
        # We use DeepCausalDiscovery as the engine
        config = {'hidden_dim': 64, 'max_iter': 50}
        active_agent = ActiveCausalAgent(DeepCausalDiscovery, config)
        
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy", "models_loaded": scm_model is not None}

@app.get("/graph", response_model=GraphResponse)
def get_graph():
    if scm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    nodes = list(scm_model.graph.nodes)
    edges = list(scm_model.graph.edges)
    return {"nodes": nodes, "edges": edges}

@app.post("/active/propose")
def propose_experiment(request: ActiveLearningRequest):
    """
    Uses the Active Agent to propose the next most informative intervention.
    """
    if active_agent is None:
        raise HTTPException(status_code=503, detail="Active Agent not initialized")
        
    df = pd.DataFrame(request.current_data)
    
    # Update beliefs (this might take time, better in background for real prod)
    active_agent.update_belief(df)
    
    target_node = active_agent.propose_intervention()
    
    if target_node:
        return {
            "message": "Uncertainty detected.",
            "proposal": f"Perform intervention do({target_node}=x)",
            "target_node": target_node,
            "reason": "High variance in edge existence probabilities."
        }
    else:
        return {
            "message": "Graph structure is stable.",
            "proposal": None
        }

@app.post("/meta/adapt")
def adapt_model(request: AdaptRequest):
    """
    Adapts the SCM to a new environment (Few-Shot Learning).
    """
    if meta_scm is None:
        raise HTTPException(status_code=503, detail="Meta-Learner not loaded")
        
    df = pd.DataFrame(request.new_data)
    
    # Preprocess
    if preprocessor:
        df_scaled = preprocessor.transform(df)
        tensor = torch.tensor(df_scaled.values).float()
    else:
        tensor = torch.tensor(df.values).float()
        
    # Reptile Step (or just simple adaptation returning new model state)
    # Here we update the base model for the session (Online Learning)
    # Or return a temporary model ID. For simplicity, we update base.
    
    meta_scm.reptile_step(tensor, k=request.steps)
    
    return {"message": "Model adapted to new environment.", "steps": request.steps}

@app.post("/intervene")
def intervene(request: InterventionRequest):
    if scm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    try:
        samples = scm_model.intervene(request.intervention, request.n_samples)
        result = pd.DataFrame(samples.numpy(), columns=scm_model.nodes).to_dict(orient="records")
        return result
    except Exception as e:
        logger.error(f"Intervention error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
