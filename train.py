import os
import random
import yaml
import argparse
import logging
from collections import defaultdict

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, f1_score

from deadlock_gnn.data.generator import generate_rag
from deadlock_gnn.data.converter import convert_to_pyg_data
from deadlock_gnn.models.rgcn_model import DeadlockRGCN
from deadlock_gnn.models.sage_model import DeadlockGNN

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_warmup_cosine_schedule(optimizer, warmup_epochs, total_epochs):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        progress = float(epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        import math
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, loader, optimizer, criterion, device, is_rgcn):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        if is_rgcn:
            out = model(data.x, data.edge_index, data.edge_type, data.batch).view(-1)
        else:
            out = model(data.x, data.edge_index, data.batch).view(-1)
            
        loss = criterion(out, data.y)
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        total_loss += loss.item()
        preds = torch.sigmoid(out).round()
        correct += (preds == data.y).sum().item()
        total += len(data.y)
    return total_loss / len(loader), correct / total

@torch.no_grad()
def evaluate(model, loader, device, is_rgcn):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
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
    return all_labels, all_preds, all_probs

def main():
    parser = argparse.ArgumentParser(description="Train DeadlockGNN")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    seed = config.get("seed", 42)
    random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # 1. Dataset generation
    logging.info("Generating synthetic RAG dataset...")
    dataset = []
    labels = []
    target_per_class = config.get("target_per_class", 1500)
    total_graphs = config.get("total_graphs_to_try", 8000)
    num_deadlocked = 0
    max_capacity = config.get("max_capacity", 3)

    for _ in range(total_graphs):
        p_count = random.randint(4, 12)
        r_count = random.randint(3, 10)
        g = generate_rag(p_count, r_count, max_capacity=max_capacity)
        if len(g.edges) == 0: continue
        
        data = convert_to_pyg_data(g, 0)
        label = data.y.item()
        
        if label == 1.0 and num_deadlocked >= target_per_class:
            continue
        if label == 0.0 and (len(dataset) - num_deadlocked) >= target_per_class:
            continue
            
        dataset.append(data)
        labels.append(label)
        num_deadlocked += int(label)

    total = len(dataset)
    logging.info(f"Total valid graphs: {total} (Deadlocked: {num_deadlocked}, Safe: {total - num_deadlocked})")

    # Stratified split 70/15/15
    train_ds, temp_ds, train_labels, temp_labels = train_test_split(
        dataset, labels, test_size=(config["val_size"] + config["test_size"]), stratify=labels, random_state=seed)
    
    val_ratio = config["val_size"] / (config["val_size"] + config["test_size"])
    val_ds, test_ds, _, _ = train_test_split(temp_ds, temp_labels, test_size=(1-val_ratio), stratify=temp_labels, random_state=seed)

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=config["batch_size"], shuffle=False)

    pos_weight = torch.tensor([(len(train_ds) - sum(train_labels)) / max(sum(train_labels), 1)], dtype=torch.float).to(device)

    # 2. Model setup
    model_type = config.get("model_type", "rgcn")
    is_rgcn = (model_type == "rgcn")
    
    if is_rgcn:
        model = DeadlockRGCN(in_channels=8, hidden_channels=config["hidden_channels"], num_relations=2, dropout=config["dropout"]).to(device)
    else:
        model = DeadlockGNN(in_channels=8, hidden_channels=config["hidden_channels"], dropout=config["dropout"]).to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    epochs = config["epochs"]
    scheduler = get_warmup_cosine_schedule(optimizer, warmup_epochs=5, total_epochs=epochs)

    # 3. Training
    best_val_f1 = 0.0
    patience = config["patience"]
    patience_count = 0
    best_state = None

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device, is_rgcn)
        scheduler.step()

        val_labels, val_preds, val_probs = evaluate(model, val_loader, device, is_rgcn)
        val_f1 = f1_score(val_labels, val_preds, zero_division=0)
        try:
            val_auc = roc_auc_score(val_labels, val_probs)
        except ValueError:
            val_auc = 0.0

        if epoch % 5 == 0 or epoch == 1:
            logging.info(f"Epoch {epoch:03d} | Loss {train_loss:.4f} | Train Acc {train_acc*100:.1f}% | Val F1 {val_f1*100:.1f}% | Val AUC {val_auc:.3f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logging.info(f"Early stopping at epoch {epoch} (best val F1: {best_val_f1*100:.1f}%)")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Save model and config
    save_path = f"deadlock_{model_type}_best.pt"
    torch.save({
        "model_state": model.state_dict(),
        "config": config,
    }, save_path)
    logging.info(f"Model saved to {save_path}")

    # Also save the test set for evaluate.py
    torch.save(test_ds, "test_dataset.pt")

if __name__ == "__main__":
    main()
