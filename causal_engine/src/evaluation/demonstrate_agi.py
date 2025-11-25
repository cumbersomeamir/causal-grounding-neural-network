import torch
import pickle
from ..causal_graph.scm import SCM
from ..causal_graph.structure_learning import load_graph
from ..reasoning.counterfactual import CounterfactualEngine
from ..utils.logger import get_logger

logger = get_logger(__name__)

def demo_ag_reasoning():
    print("\n" + "="*60)
    print("AGI CAUSAL REASONING ENGINE: Counterfactuals")
    print("="*60)
    
    try:
        graph = load_graph("artifacts/graph.pkl")
        scm = SCM(graph)
        scm.load_state_dict(torch.load("artifacts/scm_model.pth"))
        scm.eval()
        
        with open("artifacts/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
            
    except Exception as e:
        print(f"Error loading system: {e}")
        return

    engine = CounterfactualEngine(scm, preprocessor)

    # Scenario from Physics:
    # We observed a shot:
    # Angle=45, Wind=10 (Storm), Drag=5, Velocity=50 -> Distance = 230 (approx)
    # Wait, High Wind -> High Angle.
    # Let's say we observed a shot with Angle=70.
    # Angle=70 implies Wind ~ 13. Distance was ~97m.
    
    observation = {
        'Wind': 13.3,
        'Drag': 6.6,
        'Angle': 70.0,
        'Velocity': 50.0,
        'Distance': 97.0
    }
    
    print("\n[Observation]")
    print(f"We saw a cannon fired at 70 degrees during a storm (Wind=13.3).")
    print(f"The ball traveled only 97.0 meters.")
    
    # Query: "What if the wind had been calm (Wind=0)?"
    # This is a Counterfactual. We keep the "Angle" mechanism noise (maybe the pilot errored?)
    # But we change the Wind input.
    
    # Actually, in our graph Wind -> Angle.
    # If we intervene on Wind (do(Wind=0)), then Angle changes too!
    # Angle = 30 + 3*Wind + noise.
    # If Wind=0, Angle becomes 30 + noise.
    # And Drag becomes 0 + noise.
    # Distance should improve massively.
    
    intervention = {'Wind': 0.0}
    
    explanation = engine.explain(observation, intervention, "Distance")
    print("\n[AGI Reasoning]")
    print(explanation)
    
    # Query 2: "What if we kept the storm (Wind=13.3) but forced the Angle to 45?"
    intervention2 = {'Angle': 45.0}
    explanation2 = engine.explain(observation, intervention2, "Distance")
    print("\n[AGI Reasoning 2]")
    print(explanation2)

if __name__ == "__main__":
    demo_ag_reasoning()

