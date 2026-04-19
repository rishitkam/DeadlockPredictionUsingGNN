# RAG Simulated Dataset Generator

A fast, highly modular parallel generator for OS Resource Allocation Graphs.
It runs full simulations (Poisson-arriving processes with hold-cycles and queues), checks their resulting structural graph architectures, assigns ground-truth deadlock values (WFG cycles), and dumps `.pt` formats out compatible with Graph Convolution Networks.

## Running

```bash
python dataset_generator/scripts/generate_dataset.py --config dataset_generator/config/config.yaml
```

Modify the config file strictly to tweak process arrival frequencies and capacity bounds!
