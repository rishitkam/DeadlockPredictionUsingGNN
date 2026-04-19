# deadlock_gnn/models/temporal_rgcn_gru.py
"""
Temporal RGCN + GRU Hybrid Model.

Architecture:
  For each timestep t in the sequence:
      G_t → RGCN encoder → global_add_pool → graph_embedding_t  (shape: [hidden])
  Sequence of graph embeddings → GRU → final hidden state → Linear → Sigmoid
  Output: probability of deadlock at t+1
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool


class TemporalRGCNGRU(nn.Module):
    """
    Temporal OS Deadlock Predictor.
    Processes a chronological sequence of Resource Allocation Graph snapshots
    and predicts whether the next OS state will contain a deadlock.
    """

    def __init__(
        self,
        in_channels: int = 7,
        hidden_channels: int = 64,
        num_relations: int = 2,
        gru_hidden: int = 128,
        num_gru_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        # ── Spatial RGCN Encoder (shared across all timesteps) ──────────────
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.dropout = dropout

        # ── Temporal GRU over the sequence of graph embeddings ──────────────
        self.gru = nn.GRU(
            input_size=hidden_channels,
            hidden_size=gru_hidden,
            num_layers=num_gru_layers,
            batch_first=True,
            dropout=dropout if num_gru_layers > 1 else 0.0,
        )

        # ── Binary Classifier ────────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden, gru_hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden // 2, 1),
        )

    def encode_graph(self, x, edge_index, edge_type, batch):
        """RGCN encoder + global pooling → single graph embedding vector."""
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.conv3(x, edge_index, edge_type)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return global_add_pool(x, batch)  # [batch_size, hidden_channels]

    def forward(self, sequence):
        """
        Args:
            sequence: list of PyG Data objects, each with
                      x, edge_index, edge_type, batch attributes.
                      Length = temporal sequence length T.
        Returns:
            logit tensor of shape [batch_size, 1]
        """
        embeddings = []
        for graph in sequence:
            emb = self.encode_graph(
                graph.x, graph.edge_index, graph.edge_type, graph.batch
            )  # [B, H]
            embeddings.append(emb)

        # Stack: [B, T, H]
        seq_tensor = torch.stack(embeddings, dim=1)

        # GRU over time: output shape [B, T, gru_hidden]
        _, h_n = self.gru(seq_tensor)

        # Use the last layer's final hidden state: [B, gru_hidden]
        last_hidden = h_n[-1]

        return self.classifier(last_hidden)  # [B, 1]
