import streamlit as st
import networkx as nx
import torch
import os
import yaml
import numpy as np

from deadlock_gnn.data.generator import generate_rag
from deadlock_gnn.models.ensemble import hybrid_detect
from deadlock_gnn.viz.rag_plot import visualise_rag
from deadlock_gnn.models.rgcn_model import DeadlockRGCN
from deadlock_gnn.models.temporal_rgcn_gru import TemporalRGCNGRU
from deadlock_gnn.explain.shapley import shapley_attribution
from deadlock_gnn.data.converter import convert_to_pyg_data

from dataset_generator.process.process_engine import OSEngine
from dataset_generator.rag.rag_builder import RAGBuilder
from dataset_generator.converter.pyg_converter import PyGConverter

# Runtime xv6 Imports
from runtime.xv6_bridge.xv6_stream_listener import XV6StreamListener
from runtime.runtime_inference import LiveDeadlockInference

st.set_page_config(page_title="DeadlockGNN Demo", layout="wide", page_icon="🔒")

# ── Model Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_static_model():
    device = torch.device("cpu")
    if os.path.exists("deadlock_rgcn_massive.pt"):
        model_path = "deadlock_rgcn_massive.pt"
    else:
        model_path = "deadlock_rgcn_best.pt"

    if not os.path.exists(model_path):
        return None

    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    hidden = config.get("hidden_channels", 64)
    dropout = config.get("dropout", 0.3)

    model = DeadlockRGCN(in_channels=7, hidden_channels=hidden,
                          num_relations=2, dropout=dropout)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    return model

@st.cache_resource
def load_temporal_model():
    device = torch.device("cpu")
    model_path = "deadlock_temporal_model.pt"
    if not os.path.exists(model_path):
        return None
        
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    config = ckpt.get("config", {})
    hidden = config.get("hidden", 64)
    gru_hidden = config.get("gru_hidden", 128)
    
    model = TemporalRGCNGRU(in_channels=7, hidden_channels=hidden,
                            gru_hidden=gru_hidden, num_gru_layers=2, dropout=0.3)
    model.load_state_dict(ckpt["model_state"] if "model_state" in ckpt else ckpt)
    model.eval()
    return model

# ── Title ──────────────────────────────────────────────────────────────────
st.title("🔒 DeadlockGNN — Advanced OS Predictor")
st.markdown("Explore both **Static Explanatory** predictions and **Continual Temporal Sequence** predictions using our hybrid RGCN and GRU recurrent models.")

static_model = load_static_model()
temporal_model = load_temporal_model()

# ── Tabs ──────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["📊 Static Snapshot & Shapley XAI", "⏱️ Temporal Sequence Animation", "🐧 Live xv6 Monitoring"])

# ==============================================================================
# TAB 1: STATIC SNAPSHOT & XAI
# ==============================================================================
with tab1:
    col_s1, col_s2 = st.columns([1, 2.5])
    
    with col_s1:
        st.header("⚙️ Graph Parameters")
        num_procs  = st.slider("Processes", 2, 20, 6, key="sp1")
        num_res    = st.slider("Resources", 2, 15, 4, key="sr1")
        p_req      = st.slider("Request Probability", 0.0, 1.0, 0.3, key="sp_req")
        p_assign   = st.slider("Assignment Probability", 0.0, 1.0, 0.3, key="sp_ass")

        st.markdown("---")
        st.header("🔍 XAI Explanation")
        show_xai   = st.checkbox("Show Shapley Importance", value=False, key="s_xai")
        top_k      = st.slider("Top-K Explanation Nodes", 1, 10, 3, disabled=not show_xai, key="sk")
        n_mc       = st.slider("Monte Carlo Samples (T)", 10, 100, 30, disabled=not show_xai, key="smc")

        generate_static = st.button("🎲 Generate Static RAG", use_container_width=True)

    if generate_static:
        st.session_state["static_G"] = generate_rag(num_procs, num_res, p_req, p_assign)
        st.session_state.pop("shapley_scores", None)

    if "static_G" in st.session_state:
        G = st.session_state["static_G"]
        
        with col_s2:
            st.subheader("🧠 Hybrid Static Detection")
            if static_model is None:
                st.error("Static model missing.")
            else:
                status, conf, cycle, prob = hybrid_detect(G, static_model, is_rgcn=True)
                c1, c2 = st.columns(2)
                with c1:
                    if status == "DEADLOCK":
                        st.error(f"**⚠️ DEADLOCK DETECTED** ({conf})")
                    else:
                        st.success(f"**✅ SAFE** ({conf})")
                        
                    if cycle:
                        st.info(f"🔄 Cycle: {', '.join(cycle)}")
                with c2:
                    st.metric("GNN Probability", f"{prob * 100:.1f}%")

            node_importance, shapley_map = None, {}
            if show_xai and static_model is not None:
                st.markdown("---")
                st.subheader("🎯 Shapley Attribution")
                if "shapley_scores" not in st.session_state:
                    with st.spinner("Computing Shapley values..."):
                        pyg_data = convert_to_pyg_data(G, label=0)
                        st.session_state["shapley_scores"] = shapley_attribution(static_model, pyg_data, is_rgcn=True, T=n_mc)
                phi = st.session_state["shapley_scores"]
                
                phi_np = phi.numpy()
                phi_norm = (phi_np - phi_np.min()) / (phi_np.max() - phi_np.min() + 1e-9)
                shapley_map = {list(G.nodes())[i]: float(phi_norm[i]) for i in range(len(G))}
                
                sorted_nodes = sorted(shapley_map, key=shapley_map.get, reverse=True)[:top_k]
                cols = st.columns(top_k)
                for idx, n in enumerate(sorted_nodes):
                    cols[idx].metric(f"Rank {idx+1}: {n}", f"{shapley_map[n]:.3f}")

            st.markdown("---")
            st.subheader("📊 Static Visualizer")
            fig = visualise_rag(G, highlight_nodes=cycle if cycle else [],
                                node_importance=shapley_map if shapley_map else None,
                                show_explanation=show_xai)
            st.pyplot(fig, use_container_width=True)

# ==============================================================================
# TAB 2: TEMPORAL SEQUENCE ANIMATION
# ==============================================================================
with tab2:
    t_col1, t_col2 = st.columns([1, 2.5])
    
    with t_col1:
        st.header("⏱️ Sequence Generation")
        seq_len = st.slider("Time-Steps to Simulate", 3, 10, 8)
        interval = st.slider("OS Ticks per Step", 1, 10, 5)
        
        generate_temporal = st.button("🌌 Simulate Flow of Time", use_container_width=True)
        
    if generate_temporal:
        with st.spinner("Running core OS Scheduler..."):
            try:
                # Load structural environment
                config = yaml.safe_load(open("dataset_generator/config/config.yaml"))
                engine = OSEngine(config)
                
                # Warmup
                for _ in range(20): engine.step()
                
                rag_seq = []
                pyg_seq = []
                
                for _ in range(seq_len):
                    for _ in range(interval): engine.step()
                    nodes, edges = engine.get_system_state()
                    rag = RAGBuilder.build_from_state(nodes, edges)
                    pyg = PyGConverter.convert(rag)
                    rag_seq.append(rag)
                    pyg_seq.append(pyg)
                    
                st.session_state["rag_seq"] = rag_seq
                st.session_state["pyg_seq"] = pyg_seq
            except Exception as e:
                st.error(f"Simulation crashed due to unstable OS variables: {e}. Try generating again.")
                
    if "rag_seq" in st.session_state:
        with t_col2:
            st.subheader("🔮 Predictive Time-Series (RGCN + GRU)")
            if temporal_model is None:
                st.warning("⚠️ No Temporal Model found! Make sure you run `python train_temporal.py`")
            else:
                with torch.no_grad():
                    # Batch the sequence [G0, G1...] via a single batch element
                    out = temporal_model(st.session_state["pyg_seq"])
                    prob = torch.sigmoid(out).item()
                    
                st.metric("Predicted DEADLOCK Probability at T+1", f"{prob * 100:.2f}%")
                if prob > 0.6:
                    st.error("🚨 Recurrent Network foresees an imminent systemic crash in the next tick!")
                else:
                    st.success("✅ OS trajectory looks stable into the future.")
                    
            st.markdown("---")
            view_step = st.slider("Scrub Timeline", 1, seq_len, 1, format="Timestep T-%d" if seq_len else "Timestep %d")
            
            st.markdown(f"### Visualizing OS State at `T - {seq_len - view_step}`")
            fig_t = visualise_rag(st.session_state["rag_seq"][view_step - 1], title=f"OS Snapshot {view_step}/{seq_len}")
            st.pyplot(fig_t, use_container_width=True)
# ==============================================================================
# TAB 3: LIVE XV6 MONITORING
# ==============================================================================
with tab3:
    st.header("⚡ Live Operating System Intrusion")
    st.markdown("""
    This tab connects directly to a running **xv6-riscv kernel**. 
    It intercepts kernel-level events (locks, scheduling) via QEMU console instrumentation 
    to reconstruct the RAG in real-time and run neural inference.
    """)

    col_l1, col_l2 = st.columns([1, 2.5])

    with col_l1:
        st.subheader("📡 Connection Control")
        
        if "xv6_monitor" not in st.session_state:
            st.session_state["xv6_monitor"] = None
            st.session_state["xv6_inference"] = LiveDeadlockInference()
            st.session_state["last_graph"] = None
            st.session_state["monitor_active"] = False

        xv6_dir = st.text_input("xv6-riscv Path", value="./xv6-riscv")
        
        btn_label = "🛑 Stop Monitoring" if st.session_state["monitor_active"] else "🚀 Start xv6 Monitoring"
        if st.button(btn_label, type="primary", use_container_width=True):
            if st.session_state["monitor_active"]:
                if st.session_state["xv6_monitor"]:
                    st.session_state["xv6_monitor"].stop()
                st.session_state["monitor_active"] = False
                st.session_state["xv6_monitor"] = None
            else:
                try:
                    def on_update(G, event):
                        # This runs in background thread
                        st.session_state["last_graph"] = G.copy()
                        st.session_state["last_event"] = event

                    listener = XV6StreamListener(xv6_dir, callback=on_update)
                    listener.start()
                    st.session_state["xv6_monitor"] = listener
                    st.session_state["monitor_active"] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Failed to start xv6: {e}")

        if st.session_state["monitor_active"]:
            st.success("Monitoring system attached to QEMU stream.")
            if st.button("🔄 Force UI Refresh"):
                st.rerun()
        else:
            st.info("Monitor is currently idle.")

    with col_l2:
        if st.session_state["monitor_active"] and st.session_state["last_graph"]:
            G = st.session_state["last_graph"]
            
            # Run Inference
            static_p, temp_p = st.session_state["xv6_inference"].process_new_graph(G)
            
            st.subheader("🧠 Live Neural Predictions")
            m1, m2 = st.columns(2)
            m1.metric("Static Deadlock Risk", f"{static_p * 100:.1f}%")
            if temp_p is not None:
                m2.metric("Temporal Forecast (T+1)", f"{temp_p * 100:.1f}%")
            else:
                m2.metric("Temporal Forecast", "Warming up...")

            if static_p > 0.7 or (temp_p and temp_p > 0.7):
                st.warning("🚨 HIGH DEADLOCK RISK: Structural anomaly detected in kernel state!")

            st.markdown("---")
            st.subheader("🗺️ Live RAG Topology")
            # Using basic visualize for speed in live mode
            fig_live = visualise_rag(G, title=f"xv6 Live State (Nodes: {len(G.nodes)})")
            st.pyplot(fig_live, use_container_width=True)
            
            # Simple loop to encourage UI refresh
            import time
            time.sleep(1)
            st.rerun()
        else:
            st.info("Start the monitor to visualize live xv6 states.")
