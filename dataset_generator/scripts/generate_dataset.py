# dataset_generator/scripts/generate_dataset.py
import os
import sys
import yaml
import time
import argparse
from multiprocessing import Pool
import random

# Ensure local imports work regardless of execution directory
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from dataset_generator.process.process_engine import OSEngine
from dataset_generator.rag.rag_builder import RAGBuilder
from dataset_generator.converter.pyg_converter import PyGConverter
from dataset_generator.dataset.dataset_writer import DatasetWriter

def run_single_simulation(args):
    """
    Executes a completely isolated OS Simulation up to max ticks.
    Extracts the graph just before exit (or exactly when deadlock occurs).
    """
    gid, config = args
    # Ensure isolation randomness
    config['simulation']['seed'] = random.randint(0, 99999999) + gid
    
    engine = OSEngine(config)
    for _ in range(config['simulation']['steps']):
        engine.step()
        
    # Get final state topological snapshot
    nodes, edges = engine.get_system_state()
    rag = RAGBuilder.build_from_state(nodes, edges)
    
    # Check if empty (sometimes bursts clear entirely leaving empty OS states)
    if len(nodes) < 2:
        return gid, None
        
    try:
        pyg_data = PyGConverter.convert(rag)
        return gid, pyg_data
    except Exception as e:
        print(f"Error converting graph {gid}: {e}")
        return gid, None


def main():
    parser = argparse.ArgumentParser(description="Gen OS Dataset")
    parser.add_argument("--config", default="dataset_generator/config/config.yaml")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    dataset_size = config['generator']['dataset_size']
    num_workers = config['generator']['num_workers']
    out_dir = config['generator']['output_dir']
    
    writer = DatasetWriter(out_dir)
    
    print(f"🚀 Starting RAG Dataset Generation Pipeline")
    print(f"Target: {dataset_size} graphs | Pool size: {num_workers}")
    print(f"Output Directory: {out_dir}")
    
    start_t = time.time()
    valid_count = 0
    deadlock_count = 0
    
    # Create arguments for pool mapping
    tasks = [(i, config) for i in range(dataset_size)]
    
    with Pool(num_workers) as pool:
        for gid, data in pool.imap_unordered(run_single_simulation, tasks, chunksize=100):
            if data is not None:
                writer.write_graph(gid, data)
                valid_count += 1
                if data.y.item() == 1.0:
                    deadlock_count += 1
                
            if valid_count % 1000 == 0 and valid_count > 0:
                print(f"✓ Generated {valid_count} / {dataset_size} graphs... (Deadlocks: {deadlock_count})")
                
    elapsed = time.time() - start_t
    print("\n✅ Dataset Generation Complete!")
    print(f"Time Taken: {elapsed:.2f} seconds")
    print(f"Graphs Per Second: {valid_count / elapsed:.2f}")
    print(f"Total Graphs: {valid_count}")
    print(f"Deadlocks Simulated: {deadlock_count} ({(deadlock_count/valid_count)*100 if valid_count>0 else 0:.2f}%)")

if __name__ == "__main__":
    main()
