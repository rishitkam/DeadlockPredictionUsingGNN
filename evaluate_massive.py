# evaluate_massive.py
import os
import glob
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve
from deadlock_gnn.models.rgcn_model import DeadlockRGCN

class DiskRAGDataset(torch.utils.data.Dataset):
    def __init__(self, pt_files):
        self.pt_files = pt_files
    def __len__(self):
        return len(self.pt_files)
    def __getitem__(self, idx):
        return torch.load(self.pt_files[idx], weights_only=False)

def main():
    print("Evaluating RGCN Model on Massive Disk Dataset")
    
    # Load test split
    files = glob.glob(os.path.join("dataset", "graph_*.pt"))
    split_idx = int(0.8 * len(files))
    test_files = files[split_idx:]
    
    dataset = DiskRAGDataset(test_files)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = DeadlockRGCN(in_channels=7, hidden_channels=64, num_relations=2, dropout=0.2).to(device)
    
    try:
        ckpt = torch.load("deadlock_rgcn_massive.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    except Exception as e:
        print("Could not load massive model:", e)
        return
        
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    
    t_start = time.time()
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            probs = torch.sigmoid(out).view(-1).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(batch.y.cpu().numpy())
            
    latency = (time.time() - t_start) / len(test_files)

    print("\n--- Evaluation Results ---")
    print(f"Metrics over {len(test_files)} massive test graphs:")
    print(f"Accuracy : {accuracy_score(all_labels, all_preds)*100:.2f}%")
    print(f"Precision: {precision_score(all_labels, all_preds, zero_division=0)*100:.2f}%")
    print(f"Recall   : {recall_score(all_labels, all_preds, zero_division=0)*100:.2f}%")
    print(f"F1 Score : {f1_score(all_labels, all_preds, zero_division=0)*100:.2f}%")
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC-ROC  : {auc:.4f}")
    except:
        print("AUC-ROC  : N/A (Only one class present)")

    print(f"Latency  : {latency:.4f}s per graph")
    print()
    print("Confusion Matrix (TN FP / FN TP):")
    cm = confusion_matrix(all_labels, all_preds)
    print(cm)
    
    # Save Visual Confusion Matrix
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Massive Confusion Matrix')
    plt.colorbar()
    plt.xticks([0, 1], ['SAFE (0)', 'DEADLOCK (1)'])
    plt.yticks([0, 1], ['SAFE (0)', 'DEADLOCK (1)'])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2. else "black")
    plt.tight_layout()
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix_massive.png')
    plt.close()
    
    # Save ROC Curve
    plt.figure()
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('ROC Curve (Massive)')
        plt.savefig('roc_curve_massive.png')
    except:
        pass
        
    # Save PR Curve
    plt.figure()
    try:
        prec, rec, _ = precision_recall_curve(all_labels, all_probs)
        plt.plot(rec, prec, color='blue', lw=2)
        plt.title('Precision-Recall Curve (Massive)')
        plt.savefig('pr_curve_massive.png')
        print("Saved ROC and PR curves to roc_curve_massive.png and pr_curve_massive.png")
    except:
        pass

if __name__ == "__main__":
    main()
