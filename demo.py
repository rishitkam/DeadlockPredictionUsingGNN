import streamlit as st
import networkx as nx
import torch
import os

from deadlock_gnn.data.generator import generate_rag
from deadlock_gnn.models.ensemble import hybrid_detect
from deadlock_gnn.viz.rag_plot import visualise_rag
from deadlock_gnn.models.rgcn_model import DeadlockRGCN

@st.cache_resource
def load_model():
    device = torch.device("cpu")
    model_path = "deadlock_rgcn_best.pt"
    if not os.path.exists(model_path):
        return None
        
    ckpt = torch.load(model_path, map_location=device)
    config = ckpt.get("config", {})
    hidden = config.get("hidden_channels", 64)
    dropout = config.get("dropout", 0.5)
    
    model = DeadlockRGCN(in_channels=7, hidden_channels=hidden, num_relations=2, dropout=dropout)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

st.title("DeadlockGNN Interactive Demo")
st.markdown("### Generate and analyse Resource Allocation Graphs for Deadlocks")

col1, col2 = st.columns([1, 2])

with col1:
    st.header("Graph Parameters")
    num_procs = st.slider("Number of Processes", 2, 20, 6)
    num_res = st.slider("Number of Resources", 2, 15, 4)
    p_req = st.slider("Request Probability", 0.0, 1.0, 0.3)
    p_assign = st.slider("Assignment Probability", 0.0, 1.0, 0.3)
    
    if st.button("Generate RAG"):
        G = generate_rag(num_procs, num_res, p_req, p_assign)
        st.session_state['G'] = G

model = load_model()

with col2:
    if 'G' in st.session_state:
        G = st.session_state['G']
        
        st.subheader("Hybrid Detection Output")
        if model is None:
            st.error("No trained RGCN model found. Run `train.py` first.")
            status, conf, cycle, prob = "UNKNOWN", "N/A", [], 0.0
        else:
            status, conf, cycle, prob = hybrid_detect(G, model, is_rgcn=True)
            
            if status == "DEADLOCK":
                st.error(f"**DEADLOCK DETECTED** (Confidence: {conf})")
            elif status == "SAFE":
                st.success(f"**SAFE STATE** (Confidence: {conf})")
            else:
                st.warning(f"**UNCERTAIN** (Confidence: {conf})")
                
            st.metric("GNN Deadlock Probability", f"{prob*100:.1f}%")
            
            if cycle:
                st.info(f"Cycle Nodes: {', '.join(cycle)}")
                
        fig = visualise_rag(G, highlight_nodes=cycle if cycle else [])
        st.pyplot(fig)
    else:
        st.info("Adjust parameters and click 'Generate RAG' to begin.")
