import torch
import pandas as pd
import numpy as np
import pickle
import argparse
from ..causal_graph.scm import SCM
from ..causal_graph.structure_learning import load_graph
from ..models.baseline_transformer import BaselineTransformer
from ..utils.logger import get_logger

logger = get_logger(__name__)

def demonstrate(scm_path="artifacts/scm_model.pth", 
                baseline_path="artifacts/baseline_model.pth", 
                graph_path="artifacts/graph.pkl",
                preprocessor_path="artifacts/preprocessor.pkl"):
    
    print("\n" + "="*50)
    print("DEMONSTRATING CAUSAL REASONING vs CORRELATION")
    print("="*50 + "\n")

    # 1. Load Models
    graph = load_graph(graph_path)
    scm = SCM(graph)
    scm.load_state_dict(torch.load(scm_path))
    scm.eval()
    
    # Need to determine input dim from graph nodes
    input_dim = len(graph.nodes)
    baseline = BaselineTransformer(input_dim)
    baseline.load_state_dict(torch.load(baseline_path))
    baseline.eval()
    
    with open(preprocessor_path, "rb") as f:
        preprocessor = pickle.load(f)

    # 2. The "Novel Question" / Scenario
    # We want to know the effect of setting X=3.0 on Z.
    # The "World" naturally has C driving X.
    # By setting X=3, we break the C -> X link.
    
    target_x_raw = 3.0
    
    # We need to handle normalization carefully.
    # The models operate in scaled space.
    # We must transform our intervention value into scaled space,
    # perform the intervention, and transform the result back.
    
    # Get mean/scale for X from preprocessor
    # Columns are likely ['C', 'X', 'Y', 'Z'] in alphabetical order or file order
    # We need to be sure of column order.
    cols = preprocessor.columns.tolist()
    x_idx = cols.index('X')
    z_idx = cols.index('Z')
    
    x_mean = preprocessor.scaler.mean_[x_idx]
    x_scale = preprocessor.scaler.scale_[x_idx]
    
    target_x_scaled = (target_x_raw - x_mean) / x_scale
    
    print(f"Scenario: Intervention do(X={target_x_raw})")
    print(f"Scaled Intervention Value: {target_x_scaled:.4f}\n")

    # 3. Causal Model Intervention (The "Do" Operator)
    # We intervene on the SCM.
    intervention = {'X': target_x_scaled}
    
    # We generate many samples to approximate the expectation
    n_samples = 1000
    with torch.no_grad():
        # The SCM intervention logic handles the graph mutability (breaking parents of X)
        # internally or via the intervene method we wrote.
        # Our SCM.intervene method sets the value and propagates.
        # Crucially, it does NOT look at C's value to determine X. It uses the forced value.
        # For downstream nodes (Y, Z), it uses the structural equations.
        
        # Note: SCM.intervene in our implementation samples root nodes (C) from their marginals
        # if they are not intervened on. This is correct for do(X). C is untouched.
        
        scm_samples_scaled = scm.intervene(intervention, n_samples)
        
    # Inverse transform to get real-world Z values
    scm_samples_raw = preprocessor.scaler.inverse_transform(scm_samples_scaled.numpy())
    scm_z_mean = scm_samples_raw[:, z_idx].mean()
    
    # 4. Baseline Model Prediction (Correlation/Conditioning)
    # The baseline model (Transformer/MLP) takes features as input and predicts.
    # To ask it "What if X=3?", we essentially feed it an input where X=3.
    # However, a standard predictive model is trained on P(Y, Z | X, C).
    # If we just feed it X=3, what do we provide for C?
    # In a real "prediction" scenario given only X, we would implicitly marginalize C or infer C.
    # But our baseline takes ALL features as input (reconstruction/denoising task style) 
    # or predicts target from features.
    # Wait, our BaselineTransformer implementation outputs 'input_dim' (reconstruction).
    # To properly compare, we should see what the baseline reconstructs Z as,
    # given X=3 and... what for C?
    
    # If we just put in mean for C (0), the baseline might output something specific.
    # But strictly speaking, a correlational model seeing X=3 "assumes" the context C 
    # that usually accompanies X=3.
    # Our simple Transformer baseline doesn't explicitly infer C from X.
    # So let's construct a test batch where X=3 and C follows the *correlation* pattern.
    # X = 0.8 C => C = X / 0.8.
    # This simulates "Conditional Expectation" E[Z | X=3].
    
    # We will construct inputs where X is set to target, and C is set to what it "should" be
    # if we just observed X (Correlation).
    
    inferred_c_raw = target_x_raw / 0.8
    
    # We create a batch approximating this conditional distribution
    # Actually, let's just show what the model predicts if we give it the "Interventional" input 
    # (X=3, C=random) vs the "Correlational" input (X=3, C=inferred).
    # A robust Causal model should handle X=3, C=random correctly (because X doesn't depend on C in intervention).
    # The Baseline might be confused or just map inputs.
    
    # Better comparison:
    # We want to check E[Z | do(X)].
    # SCM does this by simulating the graph.
    # Baseline just maps Input -> Output.
    # If we feed the Baseline (X=3, C=0) [mean C], it gives one answer.
    # If we feed the Baseline (X=3, C=3.75) [inferred C], it gives another.
    # The "Correct" causal answer relies on C distribution being unchanged P(C).
    # So we should feed the Baseline: X=3 (fixed), C ~ N(0,1) (marginal).
    # And see if it predicts Z correctly.
    # Spoiler: The Baseline learned correlations. It learned Z = 2Y - X and Y = 1.5X + 0.5C.
    # So Z = 2(1.5X + 0.5C) - X = 3X + C - X = 2X + C.
    # If the Baseline learned this function $Z = f(X,C) = 2X + C$ perfectly, 
    # then providing X=3 and C~N(0,1) would actually give roughly 2(3) + 0 = 6.
    # So a perfect regressor on all parents *can* answer causal questions if adjustment set is valid.
    # BUT, usually baselines learn shortcuts or we don't observe C.
    # In our dataset, we observe C.
    
    # To make this "Ilya" style proving:
    # We need to show that the SCM learned the structure, whereas the baseline is just a function approximator.
    # Let's look at the Graph structure learned.
    
    print("--- Structure Learning Validation ---")
    print(f"Learned Edges: {scm.graph.edges}")
    expected_edges = [('C', 'X'), ('C', 'Y'), ('X', 'Y'), ('X', 'Z'), ('Y', 'Z')]
    print(f"Expected Edges (approx): {expected_edges}")
    
    print("\n--- Quantitative Result ---")
    print(f"True Causal Effect E[Z | do(X={target_x_raw})]: ~6.0")
    print(f"Correlational Expectation E[Z | X={target_x_raw}]: ~9.75")
    print(f"SCM Prediction (Interventional): {scm_z_mean:.4f}")
    
    # Let's check what the baseline says if we treat it as a "do" engine
    # i.e. Feed X=3, C sampled from prior (since do(X) shouldn't change C)
    c_samples = np.random.normal(0, 1, n_samples)
    c_mean = preprocessor.scaler.mean_[cols.index('C')]
    c_scale = preprocessor.scaler.scale_[cols.index('C')]
    c_scaled = (c_samples - c_mean) / c_scale
    
    # Prepare batch for baseline
    # We need to fill Y as well? 
    # The baseline is a reconstruction transformer. It takes X, C, Y, Z and reconstructs.
    # Or if it's a predictor, it predicts Z from X, C, Y.
    # Wait, our BaselineTransformer takes [batch, input_dim] and outputs [batch, input_dim].
    # It's an autoencoder style.
    # To predict Z, we mask Z? Or we just feed it.
    # This architecture isn't a direct predictor.
    # Let's modify the Baseline usage to be fair. 
    # We will feed it X, C (sampled), and MASK Y and Z (set to 0 or mean) 
    # and see if it can hallucinate the correct Z.
    # This is how BERT works.
    
    # Construct batch
    batch_raw = np.zeros((n_samples, 4))
    # Set C
    batch_raw[:, cols.index('C')] = c_samples
    # Set X
    batch_raw[:, cols.index('X')] = target_x_raw
    # Set Y to mean (0) - we don't know Y in interventional setting initially
    # Set Z to mean (0)
    
    # Scale
    batch_scaled = preprocessor.transform(pd.DataFrame(batch_raw, columns=cols)).values
    batch_tensor = torch.tensor(batch_scaled).float()
    
    with torch.no_grad():
        baseline_out = baseline(batch_tensor)
    
    baseline_out_raw = preprocessor.scaler.inverse_transform(baseline_out.numpy())
    baseline_z_mean = baseline_out_raw[:, z_idx].mean()
    
    print(f"Baseline Prediction (Reconstruction given X, C): {baseline_z_mean:.4f}")
    
    print("\n" + "="*50)
    print("CONCLUSION")
    print("="*50)
    error_scm = abs(scm_z_mean - 6.0)
    error_corr = abs(scm_z_mean - 9.75)
    
    if error_scm < 1.0:
        print("SUCCESS: The Causal Engine correctly estimated the interventional quantity.")
        print("It effectively 'imagined' a world where X was modified without changing C,")
        print("and propagated that change to Y and Z through the learned graph.")
    else:
        print("FAILURE: The Causal Engine failed to capture the true effect.")

    if abs(baseline_z_mean - 6.0) > abs(scm_z_mean - 6.0):
         print(f"The SCM outperformed the Baseline by reducing error by {abs(baseline_z_mean - 6.0) - abs(scm_z_mean - 6.0):.4f}")

if __name__ == "__main__":
    demonstrate()

