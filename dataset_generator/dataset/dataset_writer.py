# dataset_generator/dataset/dataset_writer.py
import os
import torch
from torch_geometric.data import Data

class DatasetWriter:
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
    def write_graph(self, gid: int, data: Data):
        """Saves a standalone PyG Data object."""
        path = os.path.join(self.output_dir, f"graph_{gid:06d}.pt")
        torch.save(data, path)
