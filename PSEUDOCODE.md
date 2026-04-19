# DeadlockGNN — Complete System Pseudocode

> **Model Performance**: Temporal Model — Accuracy: 93.74% | F1: 94.38% | AUC-ROC: 0.9831

---

## PART 1: CORE DATA STRUCTURE — The Resource Allocation Graph (RAG)

```
DEFINE RAG as a Directed Graph G = (V, E, R_edge, node_features)
    V = Set of nodes where each node n has:
        n.type ∈ {PROCESS, RESOURCE}
        n.features ← 7-dimensional vector
    E = Set of directed edges where each edge e has:
        e.type ∈ {REQUEST, ASSIGNMENT}
    R_edge = Set of relation types = {REQUEST=0, ASSIGNMENT=1}

DEFINE node_feature_vector(n):
    RETURN [
        is_process(n),                          // 1 if Process, 0 if Resource
        is_resource(n),                         // 0 if Process, 1 if Resource
        degree(n) / max_degree_in_graph,        // Normalised connectivity
        in_degree(n) / (out_degree(n) + 1),     // In-Out ratio
        count_request_edges(n),                 // How many things this node wants
        count_assignment_edges(n),              // How many things this node holds
        num_holders(n) / capacity(n)            // Resource utilisation (0 for processes)
    ]
```

---

## PART 2: WAIT-FOR GRAPH (WFG) & DEADLOCK GROUND TRUTH

```
ALGORITHM Build_WaitFor_Graph(RAG G):
    INPUT:  RAG graph G with process and resource nodes
    OUTPUT: WFG — a graph over processes only

    WFG ← empty directed graph

    FOR each resource R in G.nodes WHERE R.type == RESOURCE:
        holders ← {P : edge(R → P) exists in G}   // Processes holding R
        waiters ← {P : edge(P → R) exists in G}   // Processes waiting for R

        FOR each waiter W in waiters:
            FOR each holder H in holders:
                IF W ≠ H:
                    WFG.add_edge(W → H)   // W is blocked by H
    RETURN WFG


ALGORITHM Detect_Deadlock(WFG):
    INPUT:  WFG — process-only directed graph
    OUTPUT: (is_deadlocked: BOOL, cycle_nodes: LIST)

    visited ← empty set
    recursion_stack ← empty set
    cycle_nodes ← empty list

    FUNCTION DFS(node):
        ADD node to visited
        ADD node to recursion_stack

        FOR each neighbour N of node in WFG:
            IF N not in visited:
                result ← DFS(N)
                IF result ≠ NONE:
                    RETURN result   // Cycle found deeper in DFS
            ELSE IF N in recursion_stack:
                RETURN [node, N]   // Back-edge found = CYCLE!

        REMOVE node from recursion_stack
        RETURN NONE

    FOR each process P not in visited:
        result ← DFS(P)
        IF result ≠ NONE:
            RETURN (TRUE, result)

    RETURN (FALSE, [])
```

---

## PART 3: SYNTHETIC DATASET GENERATION

```
ALGORITHM Generate_Static_Dataset(target_per_class, total_attempts):
    INPUT:  target_per_class = 1500 (balanced)
    OUTPUT: dataset of PyG graph objects with labels

    dataset ← []
    deadlock_count ← 0
    safe_count ← 0

    FOR attempt = 1 TO total_attempts:
        // Randomly sample OS configuration
        num_processes ← random_int(4, 12)
        num_resources ← random_int(3, 10)

        // Build a random RAG
        G ← Build_Random_RAG(num_processes, num_resources)

        IF G has no edges: CONTINUE

        // Compute ground-truth label using WFG
        WFG ← Build_WaitFor_Graph(G)
        is_dl, _ ← Detect_Deadlock(WFG)
        label ← 1.0 IF is_dl ELSE 0.0

        // Class balance enforcement
        IF label == 1.0 AND deadlock_count >= target_per_class: CONTINUE
        IF label == 0.0 AND safe_count >= target_per_class: CONTINUE

        // Convert to PyG Data object
        pyg_data ← Convert_To_PyG(G, node_features, edge_types, label)
        APPEND pyg_data TO dataset

        IF label == 1.0: INCREMENT deadlock_count
        ELSE:            INCREMENT safe_count

    RETURN dataset


ALGORITHM Generate_Temporal_Dataset(num_sequences, seq_length=8):
    INPUT:  Number of temporal sequences to generate
    OUTPUT: List of sequence files each containing 8 RAG snapshots + future label

    FOR seq = 1 TO num_sequences:
        os_simulator ← Initialise_OS_Simulator(
            n_resources=15,
            max_capacity=5,
            poisson_lambda=0.4,    // Avg process arrivals per tick
            burst_probability=0.05
        )

        // Warmup the simulator
        FOR tick = 1 TO 20:
            os_simulator.step()

        snapshots ← []

        // Collect seq_length + 1 snapshots (last one is the label source)
        FOR t = 1 TO seq_length + 1:
            FOR _ = 1 TO snapshot_interval (=5 ticks):
                os_simulator.step()

            G_t ← os_simulator.get_current_RAG()
            pyg_t ← Convert_To_PyG(G_t)
            APPEND pyg_t TO snapshots

        // Label = is the LAST snapshot deadlocked?
        final_G ← last element of snapshots
        WFG ← Build_WaitFor_Graph(final_G)
        label, _ ← Detect_Deadlock(WFG)

        sequence_sample ← {
            "graphs": snapshots[0..seq_length-1],  // 8 past states
            "label":  float(label)                  // Future deadlock?
        }

        SAVE sequence_sample TO disk
```

---

## PART 4: THE RELATIONAL GCN (RGCN) — STATIC MODEL

```
MODEL DeadlockRGCN:
    // Handles TWO types of edges: REQUEST and ASSIGNMENT
    LAYERS:
        conv1 ← RGCNConv(in=7,      out=64, num_relations=2)
        conv2 ← RGCNConv(in=64,     out=64, num_relations=2)
        conv3 ← RGCNConv(in=64,     out=64, num_relations=2)
        classifier ← Sequential[
            Linear(64 → 32),
            ReLU(),
            Dropout(p=0.3),
            Linear(32 → 1)
        ]

    FORWARD(node_features X, edge_index, edge_type, batch_vector):
        // Message passing — each relation type uses its OWN weights
        h1 ← ReLU( conv1(X, edge_index, edge_type) )
        h2 ← ReLU( conv2(h1, edge_index, edge_type) )
        h3 ← conv3(h2, edge_index, edge_type)

        // Pool ALL node embeddings into ONE graph-level vector
        graph_embedding ← GlobalAddPool(h3, batch_vector)

        // Apply dropout and classify
        graph_embedding ← Dropout(graph_embedding, p=0.3)
        logit ← classifier(graph_embedding)

        RETURN logit   // Use Sigmoid(logit) for probability


// Why RGCN and not plain GCN?
// RGCNConv(X, edge_index, edge_type)[i] = Σ_r Σ_j (1/c_{i,r}) * W_r * X[j]
//   where j iterates over neighbours of i via relation r
//   Each relation r has its OWN weight matrix W_r
//   This lets the model learn: "being waited upon" ≠ "waiting for"
```

---

## PART 5: THE TEMPORAL MODEL — RGCN + GRU HYBRID

```
MODEL TemporalRGCNGRU:
    // Inherits spatial understanding via Transfer Learning from DeadlockRGCN
    LAYERS:
        // Spatial encoder (shared across ALL timesteps)
        conv1, conv2, conv3 ← same RGCN layers as DeadlockRGCN
                              (weights LOADED from deadlock_rgcn_massive.pt)

        // Temporal encoder
        gru ← GRU(input_size=64, hidden_size=128, num_layers=2, batch_first=True)

        // Classifier head
        classifier ← Sequential[
            Linear(128 → 64),
            ReLU(),
            Dropout(p=0.3),
            Linear(64 → 1)
        ]

    FUNCTION encode_graph(G_t):
        // Convert a single RAG snapshot to a graph embedding vector
        h ← ReLU(conv1(G_t.x, G_t.edge_index, G_t.edge_type))
        h ← ReLU(conv2(h, G_t.edge_index, G_t.edge_type))
        h ← conv3(h, G_t.edge_index, G_t.edge_type)
        RETURN GlobalAddPool(h, G_t.batch)   // Shape: [batch_size, 64]

    FORWARD(sequence = [G_0, G_1, ..., G_7]):
        embeddings ← []

        // Step 1: Encode each RAG snapshot independently
        FOR each G_t in sequence:
            e_t ← encode_graph(G_t)     // Shape: [B, 64]
            APPEND e_t TO embeddings

        // Step 2: Stack into time-series tensor
        seq_tensor ← Stack(embeddings, dim=time)  // Shape: [B, T=8, H=64]

        // Step 3: Run GRU over the temporal sequence
        _, h_n ← gru(seq_tensor)        // h_n shape: [num_layers, B, 128]

        // Step 4: Use the FINAL hidden state of the LAST layer
        last_hidden ← h_n[-1]           // Shape: [B, 128]

        // Step 5: Classify the temporal pattern
        logit ← classifier(last_hidden)

        RETURN logit   // Sigmoid(logit) = P(deadlock at T+1)


ALGORITHM Transfer_Learning():
    // Inject spatial knowledge from static model into temporal model
    static_ckpt ← LOAD("deadlock_rgcn_massive.pt")
    static_weights ← static_ckpt["model_state"]

    // Copy conv1, conv2, conv3 weights directly
    temporal_model.conv1.load(static_weights["conv1.*"])
    temporal_model.conv2.load(static_weights["conv2.*"])
    temporal_model.conv3.load(static_weights["conv3.*"])
    // The GRU and classifier learn from scratch on top of this foundation
```

---

## PART 6: TRAINING LOOP

```
ALGORITHM Train_Static_RGCN(dataset, config):
    // Stratified 70/15/15 split
    train_set, val_set, test_set ← StratifiedSplit(dataset, ratios=[0.70, 0.15, 0.15])

    model ← DeadlockRGCN(in=7, hidden=64, relations=2, dropout=0.3)
    optimizer ← Adam(model.params, lr=1e-3, weight_decay=1e-4)

    // Handle class imbalance with weighted loss
    pos_weight ← num_negatives / num_positives
    criterion ← BCEWithLogitsLoss(pos_weight=pos_weight)

    // Warmup (5 epochs) then cosine annealing
    scheduler ← WarmupCosineSchedule(warmup=5, total=config.epochs)

    best_val_f1 ← 0.0
    patience_count ← 0

    FOR epoch = 1 TO config.epochs:
        // ── Training ──
        FOR each batch in train_set:
            logits ← model.FORWARD(batch.x, batch.edge_index, batch.edge_type, batch.batch)
            loss ← criterion(logits, batch.y)
            loss.backward()
            clip_gradients(model, max_norm=1.0)   // Prevent exploding gradients
            optimizer.step()

        // ── Validation ──
        val_f1, val_auc ← Evaluate(model, val_set)
        scheduler.step()

        // Early stopping with patience
        IF val_f1 > best_val_f1:
            best_val_f1 ← val_f1
            SAVE best model checkpoint
            patience_count ← 0
        ELSE:
            patience_count += 1
            IF patience_count >= config.patience:
                BREAK   // Stop early

    RETURN best model


ALGORITHM Train_Temporal_GRU(sequence_dataset):
    Transfer_Learning()   // Inject static RGCN weights first

    FOR epoch = 1 TO 30:
        FOR each (sequence, label) in train_loader:
            logit ← temporal_model.FORWARD(sequence)
            loss ← BCEWithLogitsLoss(logit, label)
            loss.backward()
            clip_gradients(temporal_model, max_norm=1.0)
            optimizer.step()

        // Scheduler: halve LR if val F1 stops improving for 3 epochs
        val_f1 ← Evaluate(temporal_model, val_loader)
        scheduler.step(1 - val_f1)

        IF val_f1 > best_f1:
            SAVE "deadlock_temporal_model.pt"
```

---

## PART 7: LIVE xv6 KERNEL MONITORING

```
ALGORITHM Kernel_Instrumentation (runs INSIDE xv6 kernel):
    // Injected into kernel/spinlock.c — fires on every lock event

    FUNCTION on_lock_acquire(lock, current_process):
        IF lock.name starts with identifiable prefix:   // Filter safe locks only
            IF current_process.pid > 0:
                PRINT "[GNN_TRACE] LOCK_ACQUIRE P{pid} R{lock.name} {ticks}"

    FUNCTION on_lock_release(lock, current_process):
        IF is_safe_lock(lock) AND current_process.pid > 0:
            PRINT "[GNN_TRACE] LOCK_RELEASE P{pid} R{lock.name} {ticks}"

    // Injected into kernel/proc.c — fires on every process event
    ON process_create(p):  PRINT "[GNN_TRACE] PROCESS_CREATE P{p.pid} - {ticks}"
    ON process_exit(p):    PRINT "[GNN_TRACE] PROCESS_EXIT P{p.pid} - {ticks}"
    ON process_sleep(p,chan): PRINT "[GNN_TRACE] PROCESS_SLEEP P{p.pid} R{chan} {ticks}"
    ON process_wake(p,chan):  PRINT "[GNN_TRACE] PROCESS_WAKE P{p.pid} R{chan} {ticks}"


ALGORITHM Parse_Kernel_Event(raw_log_line):
    INPUT:  "[GNN_TRACE] LOCK_ACQUIRE P5 Rvirtio_disk 1642"
    OUTPUT: event dict

    PATTERN ← regex("[GNN_TRACE] (EVENT_TYPE) P(pid) (resource) (timestamp)")

    IF line matches PATTERN:
        RETURN {
            "type":      match.group(1),   // e.g., "LOCK_ACQUIRE"
            "pid":       int(match.group(2)),
            "resource":  match.group(3),
            "timestamp": int(match.group(4))
        }
    RETURN NULL


ALGORITHM Build_Live_RAG(event_stream):
    // Maintained as a live, incrementally updated NetworkX DiGraph
    G ← empty DiGraph

    ON event "PROCESS_CREATE"(pid):
        G.add_node("P{pid}", type=PROCESS)

    ON event "PROCESS_EXIT"(pid):
        REMOVE all edges connected to "P{pid}" from G
        G.remove_node("P{pid}")

    ON event "LOCK_ACQUIRE"(pid, resource):
        G.ensure_node("P{pid}", type=PROCESS)
        G.ensure_node("R_{resource}", type=RESOURCE)
        // Resource now holds process → assignment edge
        G.add_edge("R_{resource}" → "P{pid}", type=ASSIGNMENT)
        // Remove any pending request edge
        G.remove_edge_if_exists("P{pid}" → "R_{resource}")

    ON event "LOCK_RELEASE"(pid, resource):
        G.remove_edge_if_exists("R_{resource}" → "P{pid}")

    ON event "PROCESS_SLEEP"(pid, channel):
        G.ensure_node("P{pid}")
        G.ensure_node("R_{channel}")
        // Process is waiting → request edge
        G.add_edge("P{pid}" → "R_{channel}", type=REQUEST)

    ON event "PROCESS_WAKE"(pid, channel):
        G.remove_edge_if_exists("P{pid}" → "R_{channel}")

    RETURN G (continuously updated)
```

---

## PART 8: LIVE INFERENCE ENGINE

```
ALGORITHM Live_Deadlock_Inference(live_G):
    INPUT:  live_G — current NetworkX DiGraph from kernel events
    OUTPUT: (static_prob, temporal_prob)

    // ─── STATIC PREDICTION ───────────────────────────────────────────
    pyg_data ← Convert_To_PyG(live_G, label=0.0)

    // Neural Calibration: real OS has noisier features than training data
    // Clip extreme values to match synthetic training distribution
    pyg_data.x[:, col_request] ← CLAMP(pyg_data.x[:, col_request], max=10.0)
    pyg_data.x[:, col_assign]  ← CLAMP(pyg_data.x[:, col_assign],  max=10.0)

    IF pyg_data has no nodes:
        static_prob ← 0.0
    ELSE:
        batch_vector ← zeros(num_nodes)   // All nodes in one "graph"
        logit ← static_model(pyg_data.x, pyg_data.edge_index, pyg_data.edge_type, batch_vector)
        static_prob ← Sigmoid(logit)

    // ─── TEMPORAL PREDICTION ─────────────────────────────────────────
    APPEND pyg_data TO graph_buffer

    IF len(graph_buffer) > sequence_length (=8):
        REMOVE oldest graph from buffer

    IF len(graph_buffer) == sequence_length:
        batched_sequence ← [Batch(g) for g in graph_buffer]
        temporal_logit ← temporal_model(batched_sequence)
        temporal_prob ← Sigmoid(temporal_logit)
    ELSE:
        temporal_prob ← NULL   // Not enough history yet ("Warming up...")

    RETURN (static_prob, temporal_prob)
```

---

## PART 9: SHAPLEY XAI EXPLANATION

```
ALGORITHM Shapley_Attribution(model, graph G, T=30 Monte_Carlo_samples):
    INPUT:  Trained GNN model, a PyG graph, T MC sample count
    OUTPUT: Shapley value φ_i for each node i

    n ← number of nodes in G
    phi ← zeros(n)

    // Baseline: prediction with ALL nodes masked (empty graph)
    baseline_pred ← model(empty_graph)

    FOR sample = 1 TO T:
        // Pick a random subset S of nodes
        S ← random subset of {0, 1, ..., n-1}

        // Predict with S and S ∪ {i}
        FOR each node i NOT in S:
            mask_without_i ← S
            mask_with_i    ← S ∪ {i}

            pred_without ← model(G masked to mask_without_i)
            pred_with    ← model(G masked to mask_with_i)

            // Marginal contribution of node i to this coalition
            phi[i] += (pred_with - pred_without)

    phi ← phi / T   // Average over all MC samples

    // Normalise to [0, 1] for visualisation
    phi_norm ← (phi - min(phi)) / (max(phi) - min(phi) + ε)

    RETURN phi_norm   // Higher = more responsible for deadlock prediction
```

---

## PART 10: FULL SYSTEM ORCHESTRATION

```
ALGORITHM Main_System_Loop():

    // ── PHASE 1: OFFLINE TRAINING ──────────────────────────────────
    static_dataset ← Generate_Static_Dataset(target_per_class=1500)
    temporal_dataset ← Generate_Temporal_Dataset(num_sequences=50000)

    static_model ← Train_Static_RGCN(static_dataset)
    temporal_model ← Train_Temporal_GRU(temporal_dataset)
    // ↑ Temporal model inherits conv weights from static_model via Transfer_Learning()


    // ── PHASE 2: EVALUATION ────────────────────────────────────────
    Evaluate(static_model, test_set)
    // Outputs metrics, ROC curve, PR curve

    Evaluate(temporal_model, sequence_test_set)
    // Accuracy: 93.74% | F1: 94.38% | AUC-ROC: 0.9831


    // ── PHASE 3: INTERACTIVE DEMO (Streamlit) ──────────────────────
    LAUNCH streamlit_dashboard():

        // Tab 1: User generates a random RAG
        G ← Generate_Random_RAG(user_params)
        is_dl, cycle ← Detect_Deadlock(Build_WaitFor_Graph(G))
        prob ← Sigmoid(static_model(G))
        DISPLAY(G, cycle, prob)
        IF user wants XAI:
            phi ← Shapley_Attribution(static_model, G)
            HIGHLIGHT top-k nodes by Shapley value

        // Tab 2: OS time-series simulation
        sequence ← [G_0, ..., G_7] from OS_Simulator
        temporal_prob ← Sigmoid(temporal_model(sequence))
        ANIMATE sequence with scrubber
        DISPLAY prediction for T+1

        // Tab 3: Live kernel monitoring
        ON "Start Monitoring":
            CREATE FIFO pipe at /tmp/xv6_gnn_pipe
            LAUNCH xv6_qemu IN new Terminal window
                // xv6 boots and [GNN_TRACE] logs stream into FIFO
            START background_thread:
                WHILE monitoring:
                    line ← read_from_FIFO()
                    event ← Parse_Kernel_Event(line)
                    IF event valid:
                        Update_Live_RAG(event)
                        live_G ← get_current_RAG()
                        static_p, temporal_p ← Live_Deadlock_Inference(live_G)
                        UPDATE dashboard every 1 second

        ON "Stop Monitoring":
            CLOSE FIFO pipe
            STOP background_thread


// ─── THE CORE INSIGHT ──────────────────────────────────────────────────────
//
//  A deadlock can be reduced to a CYCLE DETECTION problem on a directed graph.
//  The graph (RAG) captures: who holds what, and who wants what.
//  A cycle in this graph = mutual blocking chains = deadlock.
//
//  Classical algorithms (WFG, Banker's) detect cycles but can't PREDICT them.
//  DeadlockGNN learns the STRUCTURAL SIGNATURE of near-deadlock states from
//  200,000+ synthetic examples, then applies that learned intuition to:
//    (a) A random OS state → "Is this deadlocked RIGHT NOW?"
//    (b) A sequence of 8 OS states → "Will the NEXT state be deadlocked?"
//    (c) A real running kernel → CONTINUOUSLY watch and warn in real-time.
//
// ──────────────────────────────────────────────────────────────────────────
```

---

*Performance Summary: Temporal Model trained on 50,000 OS sequences achieves 93.74% accuracy and 0.9831 AUC-ROC at 0.003s per inference — fast enough for real-time OS monitoring.*
