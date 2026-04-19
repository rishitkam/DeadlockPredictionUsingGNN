import torch

def shapley_attribution(model: torch.nn.Module, data, is_rgcn: bool = True, T: int = 50):
    """
    Monte Carlo Shapley Node Attribution for explainability.
    Formula from Convul eq. 13.
    """
    n_nodes = data.x.shape[0]
    phi = torch.zeros(n_nodes)
    device = next(model.parameters()).device
    model.eval()
    
    data = data.to(device)
    batch = torch.zeros(n_nodes, dtype=torch.long, device=device)
    
    def model_score(mask):
        masked_x = data.x.clone()
        # Set features to zero for masked out nodes
        masked_x[~mask] = 0.0
        with torch.no_grad():
            if is_rgcn:
                out = model(masked_x, data.edge_index, data.edge_type, batch).view(-1)
            else:
                out = model(masked_x, data.edge_index, batch).view(-1)
            return torch.sigmoid(out).item()

    for t in range(T):
        perm = torch.randperm(n_nodes)
        mask = torch.zeros(n_nodes, dtype=torch.bool, device=device)
        
        prev_score = model_score(mask)
        
        for v in perm:
            mask[v] = True
            new_score = model_score(mask)
            phi[v] += (new_score - prev_score)
            prev_score = new_score
            
    return phi / T
