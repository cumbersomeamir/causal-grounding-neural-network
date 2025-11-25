from fastapi import FastAPI, HTTPException
import uvicorn
import torch
import pickle
import pandas as pd
import networkx as nx
from typing import List
import os

from .api import PredictionRequest, InterventionRequest, CounterfactualRequest, GraphResponse, SampleRequest
from ..causal_graph.scm import SCM
from ..causal_graph.structure_learning import load_graph
from ..causal_graph.interventions import do_intervention
from ..utils.logger import get_logger

app = FastAPI(title="Causal Engine API", description="Microservice for Causal Inference")
logger = get_logger(__name__)

# Global state for loaded models
scm_model = None
preprocessor = None

@app.on_event("startup")
def load_models():
    global scm_model, preprocessor
    try:
        graph_path = "artifacts/graph.pkl"
        model_path = "artifacts/scm_model.pth"
        preprocessor_path = "artifacts/preprocessor.pkl"
        
        if os.path.exists(graph_path) and os.path.exists(model_path):
            graph = load_graph(graph_path)
            scm_model = SCM(graph)
            scm_model.load_state_dict(torch.load(model_path))
            scm_model.eval()
            logger.info("SCM Model loaded successfully.")
        else:
            logger.warning("Model artifacts not found. Please train the model first.")
            
        if os.path.exists(preprocessor_path):
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
    except Exception as e:
        logger.error(f"Error loading models: {e}")

@app.get("/health")
def health_check():
    return {"status": "healthy"}

@app.get("/graph", response_model=GraphResponse)
def get_graph():
    if scm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    nodes = list(scm_model.graph.nodes)
    edges = list(scm_model.graph.edges)
    return {"nodes": nodes, "edges": edges}

@app.post("/intervene")
def intervene(request: InterventionRequest):
    if scm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        samples = scm_model.intervene(request.intervention, request.n_samples)
        # Inverse transform if preprocessor is available? 
        # For simplicity, returning scaled values or raw values as produced by SCM
        # In production, we should inverse transform to original scale.
        
        result = pd.DataFrame(samples.numpy(), columns=scm_model.nodes).to_dict(orient="records")
        return result
    except Exception as e:
        logger.error(f"Intervention error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/counterfactual")
def counterfactual(request: CounterfactualRequest):
    if scm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Parse observation
        obs_df = pd.DataFrame([request.observation])
        # Transform using preprocessor if needed - assuming input is raw
        if preprocessor:
            obs_transformed = preprocessor.transform(obs_df)
            obs_tensor = torch.tensor(obs_transformed.values).float()
        else:
            obs_tensor = torch.tensor(obs_df.values).float()
            
        # For internal SCM call, we need tensor [1, n_nodes]
        # The SCM counterfactual method handles the rest
        # Note: Preprocessor scaling might complicate "do" values if they are provided in raw scale.
        # Ideally we transform intervention values too.
        
        # Simplified: Assuming raw inputs match trained scale or user handles scaling for now.
        
        cf_tensor = scm_model.counterfactual(obs_tensor, request.intervention)
        
        # Inverse transform for output
        if preprocessor:
            # We need to use inverse_transform but preprocessor might return df or array
            # Preprocessor wrapper in this codebase returns df usually but internal scaler uses array
            cf_array = preprocessor.scaler.inverse_transform(cf_tensor.numpy())
            result = pd.DataFrame(cf_array, columns=scm_model.nodes).to_dict(orient="records")
        else:
            result = pd.DataFrame(cf_tensor.numpy(), columns=scm_model.nodes).to_dict(orient="records")
            
        return result
    except Exception as e:
        logger.error(f"Counterfactual error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/sample")
def sample(request: SampleRequest):
    if scm_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    samples = scm_model.sample(request.n_samples)
    if preprocessor:
        samples_array = preprocessor.scaler.inverse_transform(samples.numpy())
        result = pd.DataFrame(samples_array, columns=scm_model.nodes).to_dict(orient="records")
    else:
        result = pd.DataFrame(samples.numpy(), columns=scm_model.nodes).to_dict(orient="records")
    return result

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

