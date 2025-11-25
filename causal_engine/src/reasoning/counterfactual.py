from ..causal_graph.scm import SCM
import torch
import numpy as np

class CounterfactualEngine:
    def __init__(self, scm: SCM, preprocessor=None):
        self.scm = scm
        self.preprocessor = preprocessor
        self.nodes = scm.nodes

    def compute_counterfactual(self, observation: dict, intervention: dict) -> dict:
        """
        Computes the counterfactual outcome given an observation and an intervention.
        """
        # 1. Preprocess Observation
        # Convert observation dict to tensor
        obs_list = [observation.get(node, 0.0) for node in self.nodes]
        # Scale if preprocessor exists
        if self.preprocessor:
            # Create DF with correct columns
            import pandas as pd
            df = pd.DataFrame([obs_list], columns=self.nodes)
            scaled = self.preprocessor.transform(df)
            obs_tensor = torch.tensor(scaled.values).float()
        else:
            obs_tensor = torch.tensor([obs_list]).float()
            
        # 2. Preprocess Intervention
        # Note: Intervention values also need scaling!
        scaled_intervention = {}
        if self.preprocessor:
             for k, v in intervention.items():
                idx = list(self.preprocessor.columns).index(k)
                mean = self.preprocessor.scaler.mean_[idx]
                scale = self.preprocessor.scaler.scale_[idx]
                scaled_intervention[k] = (v - mean) / scale
        else:
            scaled_intervention = intervention

        # 3. Run SCM Counterfactual (Abduction-Action-Prediction)
        cf_tensor = self.scm.counterfactual(obs_tensor, scaled_intervention)
        
        # 4. Inverse Transform Result
        if self.preprocessor:
            cf_vals = self.preprocessor.scaler.inverse_transform(cf_tensor.numpy())[0]
        else:
            cf_vals = cf_tensor.numpy()[0]
            
        result = {node: float(val) for node, val in zip(self.nodes, cf_vals)}
        return result

    def explain(self, observation: dict, intervention: dict, target: str):
        """
        Provides a text explanation of the causal reasoning.
        """
        cf_result = self.compute_counterfactual(observation, intervention)
        
        obs_val = observation[target]
        cf_val = cf_result[target]
        
        diff = cf_val - obs_val
        direction = "increased" if diff > 0 else "decreased"
        
        explanation = (
            f"Observed {target} was {obs_val:.2f}.\n"
            f"We simulated a world where {list(intervention.keys())[0]} was set to {list(intervention.values())[0]}.\n"
            f"In that counterfactual world, {target} would have {direction} to {cf_val:.2f}.\n"
            f"This confirms that the intervention has a causal effect of {diff:.2f} on {target}, "
            f"accounting for the specific context of this observation (e.g., hidden noise/confounders)."
        )
        return explanation
