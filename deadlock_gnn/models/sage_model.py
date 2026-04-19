import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_add_pool

class DeadlockGNN(nn.Module):
    """
    Original 3-layer GraphSAGE network for binary graph classification.
    Expects homogeneous edges.
    """
    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, dropout: float = 0.5):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, batch):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
