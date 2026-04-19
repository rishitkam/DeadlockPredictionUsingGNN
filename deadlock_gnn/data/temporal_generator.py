# deadlock_gnn/data/temporal_generator.py
"""
Generates temporal sequences of RAG graphs by simulating OS evolution over time.
Each sample: a list of PyG Data objects [G_t0, G_t1, ..., G_tN] + a label.
Label = 1 if the OS state at t+1 (after the sequence) contains a deadlock.

Serialized as a single dict .pt file:  {"graphs": [Data, ...], "label": tensor}
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import random
import time
import yaml
import torch
from multiprocessing import Pool
from torch_geometric.data import Data

from dataset_generator.process.process_engine import OSEngine
from dataset_generator.rag.rag_builder import RAGBuilder
from dataset_generator.converter.pyg_converter import PyGConverter
from dataset_generator.algorithms.wait_for_graph import detect_deadlock_wfg


def simulate_temporal_sequence(args):
    """
    Runs a full OS simulation.
    Takes snapshots every `snapshot_interval` ticks for `sequence_length` steps.
    Runs one more step and labels whether that final state is deadlocked.
    """
    gid, config = args
    seq_len = config['temporal']['sequence_length']
    interval = config['temporal']['snapshot_interval']

    # Isolate randomness per worker
    config['simulation']['seed'] = random.randint(0, 99_999_999) + gid
    engine = OSEngine(config)

    # Warm-up steps so the OS has some state
    warmup = config['simulation']['steps'] // 2
    for _ in range(warmup):
        engine.step()

    # Collect sequence snapshots
    snapshots = []
    for _ in range(seq_len):
        for _ in range(interval):
            engine.step()
        nodes, edges = engine.get_system_state()
        if len(nodes) < 2:
            return gid, None
        rag = RAGBuilder.build_from_state(nodes, edges)
        try:
            pyg = PyGConverter.convert(rag)
        except Exception:
            return gid, None
        snapshots.append(pyg)

    if len(snapshots) < seq_len:
        return gid, None

    # One more tick to determine the FUTURE label
    for _ in range(interval):
        engine.step()
    nodes_next, edges_next = engine.get_system_state()
    if len(nodes_next) < 2:
        return gid, None

    rag_next = RAGBuilder.build_from_state(nodes_next, edges_next)
    is_deadlock = detect_deadlock_wfg(rag_next)
    label = torch.tensor([1.0 if is_deadlock else 0.0], dtype=torch.float)

    return gid, {"graphs": snapshots, "label": label}


def generate_temporal_dataset(config_path: str):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    out_dir = config['temporal']['output_dir']
    n = config['temporal']['dataset_size']
    workers = config['temporal']['num_workers']
    os.makedirs(out_dir, exist_ok=True)

    print(f"🕒 Generating {n} temporal sequences (length={config['temporal']['sequence_length']}) ...")
    t_start = time.time()

    tasks = [(i, config) for i in range(n)]
    valid = 0
    deadlocks = 0

    with Pool(workers) as pool:
        for gid, result in pool.imap_unordered(simulate_temporal_sequence, tasks, chunksize=50):
            if result is not None:
                path = os.path.join(out_dir, f"seq_{gid:06d}.pt")
                torch.save(result, path)
                valid += 1
                if result["label"].item() == 1.0:
                    deadlocks += 1
            if valid % 5000 == 0 and valid > 0:
                pct = deadlocks / valid * 100
                print(f"  ✓ {valid}/{n}  (deadlock rate: {pct:.1f}%)")

    elapsed = time.time() - t_start
    pct = deadlocks / valid * 100 if valid > 0 else 0
    print(f"\n✅ Done! {valid} sequences in {elapsed:.1f}s ({valid/elapsed:.0f}/s)")
    print(f"   Deadlock rate: {deadlocks}/{valid} ({pct:.1f}%)")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="dataset_generator/config/config.yaml")
    args = parser.parse_args()
    generate_temporal_dataset(args.config)
