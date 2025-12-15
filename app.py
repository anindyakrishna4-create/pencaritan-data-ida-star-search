# File: app.py (Virtual Lab IDA* Search - VERSI PALING STABIL DAN FINAL)

import streamlit as st
import pandas as pd
import time
import matplotlib.pyplot as plt
import networkx as nx 
import random

# List global untuk menyimpan riwayat langkah
HISTORY = []

# =================================================================
# --- FUNGSI ALGORITMA IDA* SEARCH ---
# =================================================================

def search(graph, heuristic, current_node, goal_node, current_g, bound, current_path):
    """
    Fungsi rekursif Depth-First Search yang dibatasi oleh f(n) <= bound.
    Mengembalikan: (biaya_next_bound_terendah, jalur_ditemukan, final_cost)
    """
    
    global HISTORY
    
    h_score = heuristic.get(current_node, float('inf'))
    f_score = current_g + h_score

    # Catat status pengecekan simpul
    HISTORY.append({
        'action': f"DFS Check: Simpul '{current_node}'. f(n)={f_score:.2f}, g(n)={current_g:.2f}, h(n)={h_score:.2f}. Batas saat ini: {bound:.2f}.",
        'f_score': f_score,
        'expanded': current_node,
        'path': current_path,
        'g_score': current_g, 
        'status': 'Check'
    })

    # Kondisi 1: Melebihi batas (Pruning)
    if f_score > bound:
        HISTORY.append({
            'action': f"Pruning! f(n)={f_score:.2f} melebihi batas {bound:.2f}.",
            'f_score': f_score,
            'expanded': current_node,
            'path': current_path,
            'status': 'Pruned'
        })
        return f_score, None, 0 
    
    # Kondisi 2: Tujuan ditemukan
    if current_node == goal_node:
        HISTORY.append({
            'action': f"Tujuan '{goal_node}' DITEMUKAN! Biaya g(n) total: {current_g:.2f}.",
            'f_score': f_score,
            'expanded': current_node,
            'path': current_path,
            'cost': current_g, 
            'status': 'Ditemukan'
        })
        return f_score, current_path, current_g 

    # Jelajahi tetangga
    min_next_bound = float('inf')
    
    if current_node in graph:
        neighbors_sorted = sorted(graph[current_node].items(), 
                                  key=lambda item: current_g + item[1] + heuristic.get(item[0], float('inf')))

        for neighbor, cost in neighbors_sorted:
            if neighbor not in current_path:
                
                new_g = current_g + cost
                new_path = current_path + [neighbor]

                # PANGGILAN REKURSIF
                result_f, result_path, final_cost = search(graph, heuristic, neighbor, goal_node, new_g, bound, new_path)

                # Jika jalur ditemukan, kembalikan segera (Optimal)
                if result_path is not None:
                    return result_f, result_path, final_cost

                # Update batas f(n) terendah yang baru ditemukan
                min_next_bound = min(min_next_bound, result_f)

    return min_next_bound, None, 0


def ida_star_search(graph, heuristic, start_node, goal_node, all_nodes_list):
    """Fungsi utama yang mengulang (Iterative Deepening) batas f(n)."""
    global HISTORY
    HISTORY = []

    initial_h = heuristic.get(start_node, 0)
    bound = initial_h
    
    HISTORY.append({
        'action': f"IDA* Dimulai. Heuristik awal: {initial_h:.2f}. Batas Awal f(n): {bound:.2f}.",
        'f_score': bound,
        'expanded': start_node,
        'path': [start_node],
        'status': 'New Bound'
    })

    while True:
        min_next_bound, final_path, final_cost = search(graph, heuristic, start_node, goal_node, 0, bound, [start_node])
        
        # 1. Jalur Optimal Ditemukan
        if final_path is not None:
            return final_path, final_cost, HISTORY 

        # 2. Tujuan tidak terjangkau
        if min_next_bound == float('inf'):
            HISTORY.append({'action': "Semua simpul telah dieksplorasi. Tujuan tidak dapat dijangkau.", 'status': 'Gagal'})
            return None, 0, HISTORY

        # 3. Tingkatkan Batas
        bound = min_next_bound
        HISTORY.append({
            'action': f"Iterasi Selesai. Meningkatkan Batas f(n) Baru ke {bound:.2f}.",
            'f_score': bound,
            'expanded': start_node,
            'path': [start_node],
            'status': 'New Bound'
        })
        
        if len(HISTORY) > 1500: 
            st.error("Terlalu banyak iterasi. Menghentikan pencarian.")
            return None, 0, HISTORY

# =================================================================
# --- KONFIGURASI DAN STREAMLIT APP ---
# =================================================================

st.set_page_config(
    page_title="Virtual Lab: IDA* Search",
    layout="wide"
)

st.title("🌟 Virtual Lab: IDA* Search Interaktif (Iterative Deepening A*)")
st.markdown("### Memadukan Optimasi Biaya dan Efisiensi Memori")
# 
st.sidebar.header("Konfigurasi Graf, Heuristik, dan Pencarian")

# Contoh Graf
default_graph_str = """
S: A=1, B=4
A: B=2, C=5
B: C=2, D=3
C: E=3, G=4
D: G=1
E: G=1
"""
input_graph_str = st.sidebar.text_area(
    "1. Definisi Graf (Simpul: Tetangga=Biaya,...)", 
    default_graph_str, height=180
)

# Contoh Heuristik (h(n))
default_heuristic_str = """
S=7
A=6
B=4
C=2
D=1
E=1
G=0
"""
input_heuristic_str = st.sidebar.text_area(
    "2. Heuristik h(n) (Format: Simpul=Estimasi)", 
    default_heuristic_str, height=150
)

# Parsing Graf dan Heuristik (TIDAK ADA PERUBAHAN DI SINI)
try:
    graph_data = {}
    nodes = set()
    graph_edges = set()
    
    for line in input_graph_str.strip().split('\n'):
        if ':' in line:
            parent, neighbors_str = line.split(':', 1)
            parent = parent.strip()
            nodes.add(parent)
            graph_data[parent] = {}
            
            for neighbor_cost in neighbors_str.split(','):
                parts = neighbor_cost.strip().split('=')
                if len(parts) == 2:
                    neighbor = parts[0].strip()
                    cost = float(parts[1].strip()) 
                    nodes.add(neighbor)
                    graph_data[parent][neighbor] = cost
                    graph_edges.add((parent, neighbor, cost))

    all_nodes = sorted(list(nodes))
    if not all_nodes:
        st.error("Graf tidak valid atau kosong.")
        st.stop()

    heuristic_data = {}
    for line in input_heuristic_str.strip().split('\n'):
        if '=' in line:
            parts = line.split('=')
            if len(parts) == 2:
                node = parts[0].strip()
                h_value = float(parts[1].strip())
                heuristic_data[node] = h_value

    default_start = 'S'
    default_goal = 'G'
    
    start_node = st.sidebar.selectbox("Simpul Awal (Start):", all_nodes, index=all_nodes.index(default_start) if default_start in all_nodes else 0)
    goal_node = st.sidebar.selectbox("Simpul Tujuan (Goal):", all_nodes, index=all_nodes.index(default_goal) if default_goal in all_nodes else (len(all_nodes)-1))

    speed = st.sidebar.slider("Kecepatan Simulasi (detik)", 0.0, 1.0, 0.1)

except Exception as e:
    st.error(f"Kesalahan parsing data: {e}")
    st.stop()


# --- Fungsi Plot Graf (Perbaikan SyntaxError ada di sini) ---
def plot_graph(graph_edges, all_nodes_list, path_found=None, expanded_node=None, heuristic_dict=None, current_f_bound=None, g_score_dict=None):
    
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes_list)
    G.add_weighted_edges_from(graph_edges)
    
    try:
        pos = nx.spring_layout(G, seed=42) 
    except:
        pos = nx.random_layout(G)
        
    fig, ax = plt.subplots(figsize=(10, 6))

    # Kode Warna Default dan Mapping (TIDAK BERMASALAH)
    node_colors = ['#cccccc'] * len(all_nodes_list)
    node_map = {node: idx for idx, node in enumerate(all_nodes_list)}
    edge_colors = ['#888888'] * len(G.edges())
    
    # 1. Menandai Simpul yang Diekspansi/Dicek (Orange)
    if expanded_node and expanded_node in node_map:
        node_colors[node_map[expanded_node]] = '#FF9900'  # KODE WARNA LENGKAP
    
    # 2. Menandai Jalur yang Ditemukan (Hijau)
    if path_found and len(path_found) > 1:
        path_edges = list(zip(path_found, path_found[1:]))
        
        for node in path_found:
             if node in node_map:
                 node_colors[node_map[node]] = '#6AA84F' # KODE WARNA LENGKAP (HIJAU)
                 
        for i, edge in enumerate(G.edges()):
            if (edge[0], edge[1]) in path_edges:
                edge_colors[i] = '#6AA84F'

    # 3. Menandai Start (Biru) dan Goal (Ungu)
    if start_node in node_map:
        node_colors[node_map[start_node]] = '#4A86E8' # KODE WARNA LENGKAP (BIRU)
    if goal_node in node_map:
        node_colors[node_map[goal_node]] = '#8E44AD' # KODE WARNA LENGKAP (UNGU)
        
    # --- Gambar Graf ---
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1500, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2, arrows=True, arrowsize=20, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)

    # Label Biaya Tepi (Weight - Merah)
    edge_labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red', ax=ax)

    # Label F-Score (Biru)
    score_labels = {}
    for node in G.nodes():
        # Hanya tampilkan g(n) simpul yang sedang dicek atau sudah ditemukan
        g = g_score_dict.get(node) if g_score_dict and node in g_score_dict else '?'
        h = heuristic_dict.get(node, '?')
        
        # Perhitungan F-score
        f = g + h if isinstance(g, (int, float)) and isinstance(h, (int, float)) else '?'
        
        score_labels[node] = f"f:{f}, h:{h}"
        
    cost_pos = {k: [v[0], v[1] - 0.05] for k, v in pos.items()}
    nx.draw_networkx_labels(G, cost_pos, labels=score_labels, font_size=8, font_color='blue', ax=ax)

    # Tampilkan Batas F-Score
    if current_f_bound is not None and current_f_bound != float('inf'):
        ax.text(0.5, 1.05, f"BATAS F(n) SAAT INI: {current_f_bound:.2f}", 
                transform=ax.transAxes, ha="center", fontsize=12, color='darkred', weight='bold')

    ax.set_title("IDA* Search Traversal (Iterative Deepening)", fontsize=14)
    ax.axis('off')
    
    plt.close(fig) 
    return fig


# --- Visualisasi Utama ---
st.markdown("---")
st.subheader("Visualisasi IDA* Search")
st.write(f"Mencari jalur dari **{start_node}** ke **{goal_node}**.")

graph_edges_list = [(u, v, d) for u, neighbors in graph_data.items() for v, d in neighbors.items()]

if st.button("Mulai Simulasi IDA* Search"):
    
    path, cost, history = ida_star_search(graph_data, heuristic_data, start_node, goal_node, all_nodes)
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
        vis_placeholder = st.empty()
        status_placeholder = st.empty() 
    with col2:
        table_placeholder = st.empty()
    
    final_path_nodes = None
    final_cost = 0
    current_bound = heuristic_data.get(start_node, 0)
    
    g_score_tracking = {} # Melacak g_score node yang sudah dicek/dievaluasi
    
    for step, state in enumerate(history):
        status = state['status']
        action = state['action']
        
        expanded_node = state.get('expanded')
        current_path = state.get('path')
        
        # Perbarui g_score tracking
        if status == 'Check' or status == 'Ditemukan':
             # Hanya simpul yang sedang dicek yang memiliki g_score di state ini
             g_score_tracking[expanded_node] = state.get('g_score', 0)
        
        if status == 'New Bound':
            current_bound = state.get('f_score')
            # Reset g_score tracking jika iterasi baru dimulai (opsional, tapi lebih bersih)
            g_score_tracking = {start_node: 0.0} 
        elif status == 'Ditemukan':
             final_path_nodes = current_path
             final_cost = state.get('cost')

        path_to_plot = final_path_nodes if final_path_nodes else current_path
        
        fig_mpl = plot_graph(
            graph_edges_list, 
            all_nodes, 
            path_found=path_to_plot, 
            expanded_node=expanded_node,
            heuristic_dict=heuristic_data,
            current_f_bound=current_bound,
            g_score_dict=g_score_tracking # Kirim seluruh g_score yang dilacak
        )

        with vis_placeholder.container():
            st.pyplot(fig_mpl, clear_figure=True)
        
        with table_placeholder.container():
             st.markdown("##### Status Iterasi")
             df_status = pd.DataFrame([
                 {'Atribut': 'Batas F(n) Saat Ini', 'Nilai': f'{current_bound:.2f}'},
                 {'Atribut': 'Simpul Diekspansi/Dicek', 'Nilai': expanded_node if expanded_node else '-'},
                 {'Atribut': 'Jalur Saat Ini', 'Nilai': ' -> '.join(current_path) if current_path else '-'},
             ])
             st.dataframe(df
