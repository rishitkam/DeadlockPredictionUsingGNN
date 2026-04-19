import networkx as nx
import matplotlib.pyplot as plt

def visualise_rag(G: nx.DiGraph, title="Resource Allocation Graph", highlight_nodes=None, node_importance=None):
    """
    Visualises the RAG.
    - Processes: blue circles
    - Resources: red squares
    - Request edges: dashed arrows
    - Assignment edges: solid arrows
    - Highlighted nodes (cycles): orange
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Layout that puts processes on one side, resources on another, or bipartite
    processes = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'process']
    resources = [n for n, d in G.nodes(data=True) if d.get('node_type') == 'resource']
    
    pos = nx.spring_layout(G, k=0.5, seed=42)
    # Alternatively bipartite layout
    # pos = nx.bipartite_layout(G, processes)
    
    if highlight_nodes is None:
        highlight_nodes = []
        
    if node_importance is None:
        node_importance = {n: 1.0 for n in G.nodes()}
        
    p_colors = ['orange' if n in highlight_nodes else 'skyblue' for n in processes]
    r_colors = ['orange' if n in highlight_nodes else 'salmon' for n in resources]
    
    p_sizes = [500 + 1000 * node_importance.get(n, 0) for n in processes]
    r_sizes = [500 + 1000 * node_importance.get(n, 0) for n in resources]

    nx.draw_networkx_nodes(G, pos, nodelist=processes, node_color=p_colors, node_shape='o', ax=ax, node_size=p_sizes)
    nx.draw_networkx_nodes(G, pos, nodelist=resources, node_color=r_colors, node_shape='s', ax=ax, node_size=r_sizes)
    
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
    
    request_edges = [(u, v) for u, v, d in G.edges(data=True) if G.nodes[u].get('node_type') == 'process']
    assign_edges = [(u, v) for u, v, d in G.edges(data=True) if G.nodes[u].get('node_type') == 'resource']
    
    nx.draw_networkx_edges(G, pos, edgelist=request_edges, style='dashed', alpha=0.7, ax=ax, arrows=True, arrowstyle='->', arrowsize=15)
    nx.draw_networkx_edges(G, pos, edgelist=assign_edges, style='solid', alpha=0.7, ax=ax, arrows=True, arrowstyle='->', arrowsize=15)
    
    ax.set_title(title)
    ax.axis('off')
    return fig
