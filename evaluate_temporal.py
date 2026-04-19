# evaluate_temporal.py
"""
Evaluation script for the Temporal RGCN + GRU model.
Measures performance on the holdout validation split (last 20%) 
of the chronological Sequence Datasets, rendering Confusion Matrices
and AUC-ROC / Precision-Recall curves.
"""
import os
import glob
import time
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, precision_recall_curve

from deadlock_gnn.models.temporal_rgcn_gru import TemporalRGCNGRU
from train_temporal import TemporalRAGDataset, collate_temporal

def main():
    print("Evaluating Temporal RGCN+GRU Model on Sequence Dataset")
    
    # Load test split (last 20% to match training validation exactly)
    files = sorted(glob.glob(os.path.join("dataset_temporal", "seq_*.pt")))
    if not files:
        print("❌ No sequence files found in dataset_temporal/. Run temporal_generator.py first.")
        return
        
    split_idx = int(0.8 * len(files))
    test_files = files[split_idx:]
    
    dataset = TemporalRAGDataset(test_files)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_temporal)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    try:
        ckpt = torch.load("deadlock_temporal_model.pt", map_location=device, weights_only=False)
        config = ckpt.get("config", {})
        hidden = config.get("hidden", 64)
        gru_hidden = config.get("gru_hidden", 128)
        
        # Instantiate with saved parameters
        model = TemporalRGCNGRU(in_channels=7, hidden_channels=hidden, gru_hidden=gru_hidden, num_gru_layers=2, dropout=0.3).to(device)
        model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    except Exception as e:
        print(f"Could not load temporal model weights: {e}")
        return
        
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []
    
    t_start = time.time()
    with torch.no_grad():
        for sequences, labels in loader:
            sequences = [s.to(device) for s in sequences]
            labels = labels.to(device)
            
            out = model(sequences).squeeze(-1)
            probs = torch.sigmoid(out).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            
            all_preds.extend(preds)
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
    latency = (time.time() - t_start) / len(test_files)

    print("\n--- Evaluation Results ---")
    print(f"Metrics over {len(test_files)} sequential time-series samples:")
    print(f"Accuracy : {accuracy_score(all_labels, all_preds)*100:.2f}%")
    print(f"Precision: {precision_score(all_labels, all_preds, zero_division=0)*100:.2f}%")
    print(f"Recall   : {recall_score(all_labels, all_preds, zero_division=0)*100:.2f}%")
    print(f"F1 Score : {f1_score(all_labels, all_preds, zero_division=0)*100:.2f}%")
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
        print(f"AUC-ROC  : {auc:.4f}")
    except:
        auc = 0.0
        print("AUC-ROC  : N/A (Only one class present)")

    print(f"Latency  : {latency:.4f}s per sequence matrix")
    
    print("\nConfusion Matrix (TN FP / FN TP):")
    print(confusion_matrix(all_labels, all_preds))
    
    # Save ROC Curve
    plt.figure()
    try:
        from sklearn.metrics import roc_curve
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.plot(fpr, tpr, color='crimson', lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('ROC Curve (Temporal Sequences)')
        plt.savefig('roc_curve_temporal.png')
    except:
        pass
        
    # Save PR Curve
    plt.figure()
    try:
        prec, rec, _ = precision_recall_curve(all_labels, all_probs)
        plt.plot(rec, prec, color='indigo', lw=2)
        plt.title('Precision-Recall Curve (Temporal Sequences)')
        plt.savefig('pr_curve_temporal.png')
        print("\nSaved high-res metric curves to roc_curve_temporal.png and pr_curve_temporal.png")
    except:
        pass

if __name__ == "__main__":
    main()
