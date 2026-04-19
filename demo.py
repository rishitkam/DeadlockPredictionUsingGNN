import streamlit as st
import networkx as nx
import torch
import os
import numpy as np

from deadlock_gnn.data.generator import generate_rag
from deadlock_gnn.models.ensemble import hybrid_detect
from deadlock_gnn.viz.rag_plot import visualise_rag
from deadlock_gnn.models.rgcn_model import DeadlockRGCN
from deadlock_gnn.explain.shapley import shapley_attribution
from deadlock_gnn.data.converter import convert_to_pyg_data

st.set_page_config(page_title="DeadlockGNN Demo", layout="wide")

# ── Model Loader ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
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
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


# ── Page Title ────────────────────────────────────────────────────────────────
st.title("🔒 DeadlockGNN — Interactive OS Deadlock Predictor")
st.markdown(
    "Generate **Resource Allocation Graph (RAG)** snapshots from a simulated OS environment "
    "and run our **RGCN + Hybrid Ensemble** predictor. Toggle the **XAI Explanation** to see "
    "which nodes are driving the deadlock prediction via Monte Carlo Shapley Values."
)

model = load_model()

# ── Sidebar Controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Graph Parameters")
    num_procs  = st.slider("Processes",             2, 20, 6)
    num_res    = st.slider("Resources",             2, 15, 4)
    p_req      = st.slider("Request Probability",   0.0, 1.0, 0.3)
    p_assign   = st.slider("Assignment Probability",0.0, 1.0, 0.3)

    st.markdown("---")
    st.header("🔍 XAI Explanation")
    show_xai   = st.checkbox("Show Shapley Importance", value=False)
    top_k      = st.slider("Top-K Explanation Nodes", 1, 10, 3,
                            disabled=not show_xai)
    n_mc       = st.slider("Monte Carlo Samples (T)", 10, 100, 30,
                            help="Higher T = more accurate but slower",
                            disabled=not show_xai)

    generate   = st.button("🎲 Generate RAG", use_container_width=True)

# ── Generate ──────────────────────────────────────────────────────────────────
if generate:
    G = generate_rag(num_procs, num_res, p_req, p_assign)
    st.session_state["G"] = G
    # Clear cached explanation when a new graph is generated
    st.session_state.pop("shapley_scores", None)

# ── Main Panel ────────────────────────────────────────────────────────────────
if "G" in st.session_state:
    G = st.session_state["G"]

    col_left, col_right = st.columns([1, 1])

    # ── Detection results ──────────────────────────────────────────────────
    with col_left:
        st.subheader("🧠 Hybrid Detection")
        if model is None:
            st.error("No trained model found. Run `train.py` or `train_massive.py` first.")
            status, conf, cycle, prob = "UNKNOWN", "N/A", [], 0.0
        else:
            status, conf, cycle, prob = hybrid_detect(G, model, is_rgcn=True)

            if status == "DEADLOCK":
                st.error(f"**⚠️ DEADLOCK DETECTED** — Confidence: {conf}")
            elif status == "SAFE":
                st.success(f"**✅ SAFE STATE** — Confidence: {conf}")
            else:
                st.warning(f"**🔶 UNCERTAIN** — Confidence: {conf}")

            st.metric("GNN Deadlock Probability", f"{prob * 100:.1f}%")

            if cycle:
                st.info(f"🔄 Cycle Nodes: {', '.join(cycle)}")

    # ── Shapley Explanation ────────────────────────────────────────────────
    node_importance = None
    shapley_map = {}

    if show_xai and model is not None:
        with col_right:
            st.subheader("🎯 Shapley Node Explanation")

            if "shapley_scores" not in st.session_state:
                with st.spinner(f"Computing Shapley values (T={n_mc})…"):
                    try:
                        pyg_data = convert_to_pyg_data(G, label=0)
                        phi = shapley_attribution(model, pyg_data,
                                                  is_rgcn=True, T=n_mc)
                        st.session_state["shapley_scores"] = phi
                    except Exception as e:
                        st.error(f"Shapley error: {e}")
                        phi = None
            else:
                phi = st.session_state["shapley_scores"]

            if phi is not None:
                nodes = list(G.nodes())

                # Normalise scores to [0, 1]
                phi_np = phi.numpy()
                phi_min, phi_max = phi_np.min(), phi_np.max()
                rng = phi_max - phi_min if phi_max != phi_min else 1.0
                phi_norm = (phi_np - phi_min) / rng

                shapley_map = {nodes[i]: float(phi_norm[i]) for i in range(len(nodes))}
                node_importance = shapley_map

                # Top-K table
                sorted_nodes = sorted(shapley_map, key=shapley_map.get, reverse=True)[:top_k]
                st.markdown(f"**Top {top_k} most influential nodes:**")
                rows = []
                for rank, n in enumerate(sorted_nodes, 1):
                    score = shapley_map[n]
                    bar = "█" * int(score * 20)
                    rows.append({"Rank": rank, "Node": n, "Score": f"{score:.4f}",
                                 "Influence": bar})
                st.table(rows)

    # ── Graph Visualisation ────────────────────────────────────────────────
    st.subheader("📊 Resource Allocation Graph")
    legend_items = "⚪ Circle = Process   🔴 Square = Resource   "
    if show_xai:
        legend_items += "🔵 Low importance → 🟠 Medium → 🔴 High"
    st.caption(legend_items)

    fig = visualise_rag(
        G,
        title="Deadlock Prediction Explanation" if show_xai else "Resource Allocation Graph",
        highlight_nodes=cycle if cycle else [],
        node_importance=shapley_map if shapley_map else None,
        show_explanation=show_xai,
    )
    st.pyplot(fig, use_container_width=True)

else:
    st.info("👈 Adjust parameters in the sidebar and click **Generate RAG** to begin.")
