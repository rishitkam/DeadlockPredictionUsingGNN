# DeadlockGNN — Algorithm Pseudocode

---

## Algorithm 1: Deadlock Ground Truth via Wait-For Graph

The fundamental insight: a deadlock ≡ a **cycle** in the process wait-for graph.

```
Algorithm DETECT_DEADLOCK(RAG G)
  Input:  Resource Allocation Graph G = (V, E)
           where V = {processes P} ∪ {resources R}
           edges: P→R (request), R→P (assignment)
  Output: (deadlocked ∈ {true, false}, cycle_path)

  // Project RAG onto process-only Wait-For Graph
  WFG ← {}
  for each resource r ∈ G:
      holders ← { p : edge(r → p) ∈ G }
      waiters ← { p : edge(p → r) ∈ G }
      for each w ∈ waiters, h ∈ holders, w ≠ h:
          WFG ← WFG ∪ { (w → h) }   // w is blocked by h

  // DFS cycle detection
  visited, rec_stack ← ∅, ∅
  function DFS(v):
      add v to visited, rec_stack
      for each neighbour u of v in WFG:
          if u ∉ visited: result ← DFS(u)
          elif u ∈ rec_stack: return [v, u]  // back-edge = cycle
          if result ≠ ∅: return result
      remove v from rec_stack
      return ∅

  for each process p ∉ visited:
      cycle ← DFS(p)
      if cycle ≠ ∅: return (true, cycle)
  return (false, ∅)
```

---

## Algorithm 2: Node Feature Encoding

Each node is embedded into a **7-dimensional feature vector** capturing structural role, contention degree, and resource utilisation.

```
Algorithm ENCODE_NODE(v, G)
  Output: f ∈ ℝ⁷

  f[0] = 1 if v is a process,          else 0
  f[1] = 1 if v is a resource,         else 0
  f[2] = deg(v) / max_deg(G)           // normalised connectivity
  f[3] = in_deg(v) / (out_deg(v) + 1) // request pressure ratio
  f[4] = |{e ∈ G : e is REQUEST, v ∈ e}|   // how much v wants
  f[5] = |{e ∈ G : e is ASSIGNMENT, v ∈ e}| // how much v holds
  f[6] = |holders(v)| / capacity(v)    // utilisation (0 for processes)

  return f
```

---

## Algorithm 3: Relational GCN Forward Pass (Static Model)

Standard GCN ignores edge *type*. RGCN uses **separate weight matrices per relation** — critical because "waiting for" and "holding" carry opposite semantic meaning in deadlock analysis.

```
Algorithm RGCN_FORWARD(X, E, T, b)
  Input:  X ∈ ℝ^{n×7}  — node features
          E             — edge list
          T             — edge types ∈ {REQUEST=0, ASSIGNMENT=1}
          b             — batch assignment vector

  // Three relational message-passing layers
  // RGCNConv: h_i = Σ_r Σ_{j∈N_r(i)} (1/|N_r(i)|) · W_r · h_j
  H ← ReLU( RGCNConv_1(X, E, T) )    // ℝ^{n×64}
  H ← ReLU( RGCNConv_2(H, E, T) )    // ℝ^{n×64}
  H ←       RGCNConv_3(H, E, T)      // ℝ^{n×64}

  // Aggregate all node embeddings → single graph vector
  z ← Σ_{i: b_i=g} H_i  for each graph g  (GlobalAddPool)  // ℝ^{64}

  // Binary classifier
  logit ← Linear(32→1)( ReLU( Linear(64→32)(z) ) )

  return σ(logit)   // deadlock probability ∈ (0,1)
```

---

## Algorithm 4: Temporal Model — RGCN Encoder + GRU

**The key novelty**: the RGCN encoder is **shared and frozen across all timesteps** (its weights are transferred from the 200k-graph pretrained static model). Only the GRU learns temporal dynamics.

```
Algorithm TEMPORAL_FORWARD(S = [G_0, G_1, ..., G_7])
  Input:  Ordered sequence of 8 RAG snapshots

  // Step 1 — Encode each snapshot with the *shared* RGCN encoder
  for t = 0 to 7:
      e_t ← RGCN_ENCODER(G_t)    // ℝ^{64}  (same weights every t)

  // Step 2 — Stack into time-series
  Z ← [e_0; e_1; ...; e_7]      // ℝ^{8×64}

  // Step 3 — Temporal reasoning with 2-layer GRU
  _, h_T ← GRU(Z, hidden=128, layers=2)   // h_T ∈ ℝ^{128}

  // Step 4 — Forecast next OS state
  logit ← Linear(64→1)( ReLU( Linear(128→64)(h_T[-1]) ) )

  return σ(logit)   // P(deadlock at t=8)


Algorithm TRANSFER_LEARNING(static_ckpt, temporal_model)
  // Inject spatial knowledge: no relearning of graph topology
  for layer ∈ {conv1, conv2, conv3}:
      temporal_model.layer.W_r ← static_ckpt.layer.W_r  ∀ r
  // GRU and classifier head train from scratch on top of this
```

---

## Algorithm 5: Live Kernel RAG Construction

The xv6 kernel emits structured trace events. The bridge maintains an **incremental RAG** — no full-rebuild, just edge deltas on each event.

```
// Kernel side (C, inside spinlock.c / proc.c)
ON acquire(lock, pid):
    if is_monitored(lock):
        printf("[GNN_TRACE] LOCK_ACQUIRE P%d %s %d\n", pid, lock.name, ticks)

ON release(lock, pid):
    if is_monitored(lock):
        printf("[GNN_TRACE] LOCK_RELEASE P%d %s %d\n", pid, lock.name, ticks)


// Python bridge side
Algorithm UPDATE_LIVE_RAG(event, G)
  match event.type:
      "LOCK_ACQUIRE"(pid, res):
          G.add_edge(res → pid, type=ASSIGNMENT)
          G.remove_edge(pid → res)         // resolve pending request

      "LOCK_RELEASE"(pid, res):
          G.remove_edge(res → pid)

      "PROCESS_SLEEP"(pid, chan):
          G.add_edge(pid → chan, type=REQUEST)  // process now waiting

      "PROCESS_WAKE"(pid, chan):
          G.remove_edge(pid → chan)

      "PROCESS_EXIT"(pid):
          G.remove_all_edges(pid)
          G.remove_node(pid)

  return G   // single incremental update, O(1) per event
```

---

## Algorithm 6: Monte-Carlo Shapley Attribution (XAI)

Answers: *which nodes are most responsible for the predicted deadlock risk?*

```
Algorithm SHAPLEY(model, G, T=30)
  Input:  trained model, graph G with n nodes, T MC samples
  Output: φ ∈ ℝ^n (importance per node)

  φ ← 0^n
  for t = 1 to T:
      S ← random subset of nodes
      for each node i ∉ S:
          φ[i] += model(G|_{S∪{i}}) − model(G|_S)
                  // marginal contribution of node i to coalition S

  φ ← φ / T
  return normalise(φ)   // high φ[i] → node i drives the deadlock
```

---

## Core Claim

> A deadlock is a **cycle** in a directed resource-allocation graph.  
> Classical algorithms detect cycles reactively.  
> DeadlockGNN learns the *structural pre-conditions* of cycle formation from 200k synthetic OS states, then applies that knowledge proactively:  
> — **Statically**: is this graph deadlocked right now?  
> — **Temporally**: will the next OS tick be deadlocked?  
> — **Live**: continuously monitor a real running kernel.

**Results**: Temporal model — Accuracy 93.74% | F1 94.38% | AUC-ROC 0.9831 | Latency 2.9ms/sequence
