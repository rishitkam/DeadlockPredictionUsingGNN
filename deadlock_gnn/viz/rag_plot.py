import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np


def visualise_rag(
    G: nx.DiGraph,
    title: str = "Resource Allocation Graph",
    highlight_nodes=None,
    node_importance: dict = None,
    show_explanation: bool = False,
):
    """
    Visualises the RAG.
    - Processes: circles | Resources: squares
    - Request edges: dashed arrows | Assignment edges: solid arrows
    - If show_explanation=True, colors nodes by Shapley importance:
        Low  → steelblue
        Mid  → darkorange
        High → crimson
    """
    fig, ax = plt.subplots(figsize=(9, 6))

    processes = [n for n, d in G.nodes(data=True) if d.get("node_type") == "process"]
    resources = [n for n, d in G.nodes(data=True) if d.get("node_type") == "resource"]

    pos = nx.spring_layout(G, k=0.6, seed=42)

    if highlight_nodes is None:
        highlight_nodes = []
    if node_importance is None:
        node_importance = {n: 0.0 for n in G.nodes()}

    # Colormap: importance 0→1 maps lightblue→orange→red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shapley", ["steelblue", "darkorange", "crimson"]
    )

    def node_color(n, default_color):
        if show_explanation and n in node_importance:
            score = float(np.clip(node_importance[n], 0.0, 1.0))
            return mcolors.to_hex(cmap(score))
        if n in highlight_nodes:
            return "gold"
        return default_color

    p_colors = [node_color(n, "skyblue") for n in processes]
    r_colors = [node_color(n, "salmon") for n in resources]
    p_sizes = [600 + 1200 * float(np.clip(node_importance.get(n, 0.0), 0, 1)) for n in processes]
    r_sizes = [600 + 1200 * float(np.clip(node_importance.get(n, 0.0), 0, 1)) for n in resources]

    nx.draw_networkx_nodes(G, pos, nodelist=processes, node_color=p_colors,
                           node_shape="o", ax=ax, node_size=p_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=resources, node_color=r_colors,
                           node_shape="s", ax=ax, node_size=r_sizes)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=9, font_weight="bold")

    req_edges = [(u, v) for u, v, d in G.edges(data=True) if G.nodes[u].get("node_type") == "process"]
    asgn_edges = [(u, v) for u, v, d in G.edges(data=True) if G.nodes[u].get("node_type") == "resource"]

    nx.draw_networkx_edges(G, pos, edgelist=req_edges, style="dashed",
                           alpha=0.7, ax=ax, arrows=True, arrowstyle="->", arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=asgn_edges, style="solid",
                           alpha=0.7, ax=ax, arrows=True, arrowstyle="->", arrowsize=15)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axis("off")

    # Legend for explanation mode
    if show_explanation:
        sm = cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(0, 1))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, orientation="horizontal",
                            fraction=0.04, pad=0.02)
        cbar.set_label("Shapley Importance  (low → high)", fontsize=9)

    return fig
