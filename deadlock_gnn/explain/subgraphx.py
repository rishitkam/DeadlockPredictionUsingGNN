import torch
from torch_geometric.explain import Explainer, GNNExplainer, ModelConfig

def get_explainer(model: torch.nn.Module, is_rgcn: bool = True):
    """
    Returns an Explainer tool using GNNExplainer as a proxy for SubgraphX interpretation.
    To use:
    explainer = get_explainer(model)
    explanation = explainer(x, edge_index, edge_type=edge_type, batch=batch)
    """
    # GNNExplainer used as default local explainer
    model_config = ModelConfig(
        mode='binary_class',
        task_level='graph',
        return_type='raw',
    )
    
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=model_config,
    )
    
    return explainer
