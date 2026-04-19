# runtime/runtime_inference.py
import torch
import networkx as nx
from typing import List, Optional
from deadlock_gnn.models.rgcn_model import DeadlockRGCN
from deadlock_gnn.models.temporal_rgcn_gru import TemporalRGCNGRU
from deadlock_gnn.data.converter import convert_to_pyg_data

class LiveDeadlockInference:
    def __init__(self, 
                 static_model_path: str = "deadlock_rgcn_massive.pt",
                 temporal_model_path: str = "deadlock_temporal_model.pt",
                 sequence_length: int = 8):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load Static Model
        self.static_model = DeadlockRGCN(num_node_features=7, num_relations=2).to(self.device)
        checkpoint_static = torch.load(static_model_path, map_location=self.device)
        state_dict_static = checkpoint_static["model_state"] if "model_state" in checkpoint_static else checkpoint_static
        self.static_model.load_state_dict(state_dict_static)
        self.static_model.eval()

        # Load Temporal Model
        self.temporal_model = TemporalRGCNGRU(num_node_features=7, num_relations=2).to(self.device)
        checkpoint_temp = torch.load(temporal_model_path, map_location=self.device)
        state_dict_temp = checkpoint_temp["model_state"] if "model_state" in checkpoint_temp else checkpoint_temp
        self.temporal_model.load_state_dict(state_dict_temp)
        self.temporal_model.eval()

        self.sequence_length = sequence_length
        self.graph_buffer = [] # List of PyG Data objects

    def process_new_graph(self, nx_graph: nx.DiGraph):
        """
        Converts the live NetworkX graph to PyG and updates the temporal buffer.
        Returns (static_prob, temporal_prob)
        """
        from torch_geometric.data import Batch
        
        # Convert to PyG (label 0.0 as it's live inference)
        # Note: label doesn't matter for inference
        pyg_data = convert_to_pyg_data(nx_graph, label=0.0).to(self.device)
        
        # Static Prediction
        with torch.no_grad():
            # Add batch attribute for global pooling
            batch_vec = torch.zeros(pyg_data.x.size(0), dtype=torch.long, device=self.device)
            static_logits = self.static_model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, batch_vec)
            static_prob = torch.sigmoid(static_logits).item()

        # Update Buffer
        self.graph_buffer.append(pyg_data)
        if len(self.graph_buffer) > self.sequence_length:
            self.graph_buffer.pop(0)

        # Temporal Prediction
        temporal_prob = None
        if len(self.graph_buffer) == self.sequence_length:
            with torch.no_grad():
                # Temporal model expects a list of Batched snapshots (one per timestep)
                # each Batch having batch size of self.batch_size.
                # Here batch_size=1.
                batched_sequence = [Batch.from_data_list([g]) for g in self.graph_buffer]
                temporal_logits = self.temporal_model(batched_sequence)
                temporal_prob = torch.sigmoid(temporal_logits).item()

        return static_prob, temporal_prob

    def reset(self):
        self.graph_buffer = []
