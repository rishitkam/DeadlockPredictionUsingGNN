import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, global_add_pool

class DeadlockRGCN(nn.Module):
    """
    3-layer RGCN network for binary graph classification.
    Uses RGCNConv to handle relation types (request vs assignment edges).
    Based on Convul Algorithm 2.
    """
    def __init__(self, in_channels: int = 8, hidden_channels: int = 64, num_relations: int = 2, dropout: float = 0.5):
        super().__init__()
        self.conv1 = RGCNConv(in_channels, hidden_channels, num_relations=num_relations)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, num_relations=num_relations)
        self.dropout = dropout

        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, 1),
        )

    def forward(self, x, edge_index, edge_type, batch):
        x = F.relu(self.conv1(x, edge_index, edge_type))
        x = F.relu(self.conv2(x, edge_index, edge_type))
        x = self.conv3(x, edge_index, edge_type)
        x = global_add_pool(x, batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.classifier(x)
