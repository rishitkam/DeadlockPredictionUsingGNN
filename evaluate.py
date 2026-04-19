import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
import time
from torch_geometric.loader import DataLoader

from deadlock_gnn.models.rgcn_model import DeadlockRGCN
from deadlock_gnn.models.sage_model import DeadlockGNN

def evaluate_model(model_path, dataset_path="test_dataset.pt", is_rgcn=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    test_ds = torch.load(dataset_path, weights_only=False)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    hidden = config.get("hidden_channels", 64)
    dropout = config.get("dropout", 0.5)
    
    if is_rgcn:
        model = DeadlockRGCN(in_channels=8, hidden_channels=hidden, num_relations=2, dropout=dropout).to(device)
    else:
        model = DeadlockGNN(in_channels=8, hidden_channels=hidden, dropout=dropout).to(device)
        
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    
    all_preds, all_labels, all_probs = [], [], []
    
    start_time = time.perf_counter()
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            if is_rgcn:
                out = model(data.x, data.edge_index, data.edge_type, data.batch).view(-1)
            else:
                out = model(data.x, data.edge_index, data.batch).view(-1)
                
            probs = torch.sigmoid(out)
            all_probs.extend(probs.cpu().tolist())
            all_preds.extend(probs.round().cpu().tolist())
            all_labels.extend(data.y.cpu().tolist())
            
    latency = time.perf_counter() - start_time
    
    acc = accuracy_score(all_labels, all_preds)
    prec = precision_score(all_labels, all_preds, zero_division=0)
    rec = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
        pr_auc = average_precision_score(all_labels, all_probs)
    except ValueError:
        auc, pr_auc = 0.0, 0.0
    cm = confusion_matrix(all_labels, all_preds)
    
    print("\n--- Evaluation Results ---")
    print(f"Metrics over {len(test_ds)} test graphs:")
    print(f"Accuracy : {acc*100:.2f}%")
    print(f"Precision: {prec*100:.2f}%")
    print(f"Recall   : {rec*100:.2f}%")
    print(f"F1 Score : {f1*100:.2f}%")
    print(f"AUC-ROC  : {auc:.4f}")
    print(f"AUC-PR   : {pr_auc:.4f}")
    print(f"Latency  : {latency:.4f}s")
    print(f"Confusion Matrix (TN FP / FN TP):\n{cm}")
    
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve
    
    try:
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig('roc_curve.png')
        plt.close()
        
        precision, recall, _ = precision_recall_curve(all_labels, all_probs)
        plt.figure()
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        plt.legend(loc="lower left")
        plt.savefig('pr_curve.png')
        plt.close()
        
        print("Saved ROC and PR curves to roc_curve.png and pr_curve.png")
    except Exception as e:
        print(f"Could not save curves: {e}")

    return acc, prec, rec, f1

if __name__ == "__main__":
    import os
    if os.path.exists("deadlock_rgcn_best.pt"):
        print("Evaluating RGCN Model")
        evaluate_model("deadlock_rgcn_best.pt", is_rgcn=True)
    elif os.path.exists("deadlock_sage_best.pt"):
        print("Evaluating SAGE Model")
        evaluate_model("deadlock_sage_best.pt", is_rgcn=False)
    else:
        print("No model checkpoint found. Train first.")
