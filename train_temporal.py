# train_temporal.py
"""
Training pipeline for the Temporal RGCN + GRU hybrid model.
Loads sequences from dataset_temporal/ (each .pt file is a dict with
  {"graphs": [Data, ...], "label": tensor})
and trains the model to predict future deadlock probability.
"""
import os
import glob
import time
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Batch
from sklearn.metrics import f1_score, roc_auc_score, precision_score, recall_score

from deadlock_gnn.models.temporal_rgcn_gru import TemporalRGCNGRU


# ── Dataset ──────────────────────────────────────────────────────────────────
class TemporalRAGDataset(Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], weights_only=False)
        return sample["graphs"], sample["label"]


def collate_temporal(batch):
    """
    Custom collate: batch a list of (graphs_list, label) pairs.
    Returns a list of Batched PyG graphs (one per timestep) + stacked labels.
    """
    seq_len = len(batch[0][0])
    # For each timestep, collect the graphs from all samples in this batch
    batched_steps = []
    for t in range(seq_len):
        graphs_at_t = [sample[0][t] for sample in batch]
        batched_steps.append(Batch.from_data_list(graphs_at_t))
    labels = torch.stack([sample[1] for sample in batch]).squeeze(-1)
    return batched_steps, labels


# ── Training Loop ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Train Temporal RGCN+GRU Model")
    parser.add_argument('--dir', type=str, default="dataset_temporal", help="Directory containing sequence .pt files")
    parser.add_argument('--epochs', type=int, default=30, help="Number of training epochs (default 30 for transfer learning)")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size for sequences")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--gru_hidden", type=int, default=128)
    args = parser.parse_args()

    files = sorted(glob.glob(os.path.join(args.dir, "seq_*.pt")))
    if not files:
        print(f"❌ No sequence files found in {args.dir}/")
        print("   Run first: python deadlock_gnn/data/temporal_generator.py")
        return

    split = int(0.8 * len(files))
    train_files, val_files = files[:split], files[split:]
    print(f"📂 Dataset: {len(train_files)} train | {len(val_files)} val sequences")

    train_ds = TemporalRAGDataset(train_files)
    val_ds = TemporalRAGDataset(val_files)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_temporal)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_temporal)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥  Device: {device}")

    model = TemporalRGCNGRU(
        in_channels=7,
        hidden_channels=args.hidden,
        gru_hidden=args.gru_hidden,
        num_gru_layers=2,
        dropout=0.3,
    ).to(device)

    # [Transfer Learning] Inject static weights into the spatial encoder
    model.load_static_weights("deadlock_rgcn_massive.pt", device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.5)

    best_f1 = 0
    train_losses, val_f1s = [], []
    t_start = time.time()

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        total_loss = 0.0
        for sequences, labels in train_loader:
            sequences = [s.to(device) for s in sequences]
            labels = labels.to(device)
            optimizer.zero_grad()
            out = model(sequences).squeeze(-1)
            loss = criterion(out, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(train_files)

        # ── Validate ──
        model.eval()
        all_preds, all_probs, all_labels = [], [], []
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = [s.to(device) for s in sequences]
                out = model(sequences).squeeze(-1)
                probs = torch.sigmoid(out).cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_preds.extend(preds)
                all_probs.extend(probs)
                all_labels.extend(labels.numpy())

        f1 = f1_score(all_labels, all_preds, zero_division=0)
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except Exception:
            auc = 0.0
        prec = precision_score(all_labels, all_preds, zero_division=0)
        rec = recall_score(all_labels, all_preds, zero_division=0)

        scheduler.step(1 - f1)
        print(f"Epoch {epoch:03d} | Loss {avg_loss:.4f} | F1 {f1:.4f} | AUC {auc:.4f} | P {prec:.3f} | R {rec:.3f}")

        train_losses.append(avg_loss)
        val_f1s.append(f1)

        if f1 > best_f1:
            best_f1 = f1
            torch.save({"model_state": model.state_dict(), "config": vars(args)},
                       "deadlock_temporal_model.pt")

    elapsed = time.time() - t_start
    print(f"\n✅ Training complete in {elapsed:.1f}s | Best Val F1: {best_f1:.4f}")
    print("   Model saved → deadlock_temporal_model.pt")

    # ── Dashboard PNG ──
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss", color="crimson")
    plt.title("Training Loss (Temporal)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(val_f1s, label="Val F1", color="steelblue")
    plt.title("Validation F1 (Temporal)")
    plt.xlabel("Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_dashboard_temporal.png")
    print("   Dashboard → training_dashboard_temporal.png")


if __name__ == "__main__":
    main()
