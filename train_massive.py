# train_massive.py
import os
import glob
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from deadlock_gnn.models.rgcn_model import DeadlockRGCN
import time
import argparse
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

class DiskRAGDataset(torch.utils.data.Dataset):
    def __init__(self, pt_files):
        self.pt_files = pt_files
        
    def __len__(self):
        return len(self.pt_files)
        
    def __getitem__(self, idx):
        return torch.load(self.pt_files[idx], weights_only=False)

def main():
    parser = argparse.ArgumentParser(description="Train RGCN on Massive Disk Dataset")
    parser.add_argument("--dir", type=str, default="dataset", help="Directory containing .pt files")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.005)
    args = parser.parse_args()

    files = glob.glob(os.path.join(args.dir, "graph_*.pt"))
    if not files:
        print(f"No .pt files found in {args.dir}!")
        return
        
    print(f"Found {len(files)} graphs in disk. Loading indices...")
    
    # Simple split
    split_idx = int(0.8 * len(files))
    train_files = files[:split_idx]
    test_files = files[split_idx:]
    
    train_ds = DiskRAGDataset(train_files)
    test_ds = DiskRAGDataset(test_files)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Initialize model using 7-dim features
    model = DeadlockRGCN(in_channels=7, hidden_channels=64, num_relations=2, dropout=0.2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_f1 = 0
    t_start = time.time()
    
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss = criterion(out.view(-1), batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch.num_graphs
            
        avg_loss = total_loss / len(train_files)
        
        # Test Validation
        model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)
                out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
                probs = torch.sigmoid(out).view(-1).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_labels.extend(batch.y.cpu().numpy())
                
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | Val F1 {f1:.4f}")
        
        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict()}, "deadlock_rgcn_massive.pt")
            
    print(f"\nTraining completed in {time.time() - t_start:.2f}s")
    print("Best F1 on new massive dataset:", best_f1)
    print("Model perfectly saved to 'deadlock_rgcn_massive.pt'")

if __name__ == "__main__":
    main()
