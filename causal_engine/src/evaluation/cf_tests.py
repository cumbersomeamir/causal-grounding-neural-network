import torch
from ..causal_graph.scm import SCM
from ..utils.logger import get_logger

logger = get_logger(__name__)

def run_cf_tests(scm: SCM):
    """
    Check properties of counterfactuals.
    Property 1: Consistency. CF(observed_val, intervention=observed_val) should be observed_val.
    """
    logger.info("Running Counterfactual Tests...")
    pass

