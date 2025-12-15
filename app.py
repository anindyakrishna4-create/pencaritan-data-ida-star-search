# File: app.py (Virtual Lab IDA* Search - Versi Paling Stabil)

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
        'g_score': current_g, # Tambahkan g_score untuk ditampilkan di visualisasi
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
        return f_score, None, 0 # Kembali: f_score, jalur=None, cost=0
    
    # Kondisi 2: Tujuan ditemukan
    if current_node == goal_node:
        HISTORY.append({
            'action': f"Tujuan '{goal_node}' DITEMUKAN! Biaya g(n) total: {current_g:.2f}.",
            'f_score': f_score,
            'expanded': current_node,
            'path': current_path,
            'cost': current_g, # Biaya g(n) akhir
            'status': 'Ditemukan'
        })
        return f_score, current_path, current_g # Kembali: f_score, jalur=list, cost=g(n)

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

    return min_next_bound, None, 0 # Kembali: f_score, jalur=None, cost=0


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
        # Panggil fungsi search() yang kini mengembalikan 3 nilai
        min_next_bound, final_path, final_cost = search(graph, heuristic, start_node, goal_node, 0, bound, [start_node])
        
        # 1. Jalur Optimal Ditemukan
        if final_path is not None:
            # FIX: final_cost kini langsung berasal dari fungsi search()
            return final_path, final_cost, HISTORY # KEMBALI DENGAN PATH (LIST OF NODES) DAN COST (FLOAT)

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
        
        if len(HISTORY) > 1500: # Batasan aman (ditingkatkan sedikit)
            st.error("Terlalu banyak iterasi. Menghentikan pencarian.")
            return None, 0, HISTORY

    
# =================================================================
# --- KONFIGURASI DAN STREAMLIT APP ---
# (Konten di bawah ini tetap sama, hanya beberapa penyesuaian untuk float)
# =================================================================

st.set_page_config(
    page_title="Virtual Lab: IDA* Search",
    layout="wide"
)

st.title("🌟 Virtual Lab: IDA* Search Interaktif (Iterative Deepening A*)")
st.markdown("### Memadukan Optimasi Biaya dan Efisiensi Memori")

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

# Parsing Graf dan Heuristik
try:
    graph_data = {}
    nodes = set()
    graph_edges = set()
    
    # Parsing Graf
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
                    cost = float(parts[1].strip()) # Menggunakan float untuk fleksibilitas
                    nodes.add(neighbor)
                    graph_data[parent][neighbor] = cost
                    graph_edges.add((parent, neighbor, cost))

    all_nodes = sorted(list(nodes))
    if not all_nodes:
        st.error("Graf tidak valid atau kosong.")
        st.stop()

    # Parsing Heuristik
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


# --- Fungsi Plot Graf (NetworkX + Matplotlib) ---
def plot_graph(graph_edges, all_nodes_list, path_found=None, expanded_node=None, heuristic_dict=None, current_f_bound=None, g_score_dict=None):
    
    G = nx.DiGraph()
    G.add_nodes_from(all_nodes_list)
    G.add_weighted_edges_from(graph_edges)
    
    try:
        pos = nx.spring_layout(G, seed=42) 
    except:
        pos = nx.random_layout(G)
        
    fig, ax = plt.subplots(figsize=(10, 6))

    node_colors = ['#cccccc'] * len(all_nodes_list)
    node_map = {node: idx for idx, node in enumerate(all_nodes_list)}
    edge_colors = ['#888888'] * len(G.edges())
    
    # 1. Menandai Simpul yang Diekspansi/Dicek (Orange)
    if expanded_node and expanded_node in node_map:
        node_colors[node_map[expanded_node]] = '#FF9900' 
    
    # 2. Menandai Jalur yang Ditemukan (Hijau)
    if path_found and len(path_found) > 1:
        path_edges = list(zip(path_found, path_found[1:]))
        
        for node in path_found:
             if node in node_map:
                 node_colors[node_map[node]] = '#6AA
