import time
import networkx as nx
import sys
import os
import json
from avg_degree import *
import random

# get graph
def load_graph(seed=42, unvacc_rate=0.3):
    
    random.seed(seed)
    
    G = nx.Graph()
    G.NAME = "montgomery"
    
    file = open("montgomery_labels_all.txt", "r")
    lines = file.readlines()
    nodes = {}
    rev_nodes = []
    c_node=0
    
    for line in lines:
        a = line.split(",")
        u = int(a[0])
        v = int(a[1])
        
        if u in nodes.keys():
            u = nodes[u]
        else:
            nodes[u] = c_node
            rev_nodes.append(u)
            u = c_node
            c_node+=1   
    
        if v in nodes.keys():
            v = nodes[v]
        else:
            nodes[v] = c_node
            rev_nodes.append(v)
            v = c_node
            c_node+=1
        
        G.add_edge(u,v)
    
    nodes = list(G.nodes)
    for n in nodes:
        # simulate already vaccinated nodes
        if random.random()>unvacc_rate:
            G.remove_node(n)
    
    mapping = dict(zip(G, range(len(G.nodes))))
    G = nx.relabel_nodes(G, mapping) 

    return G


#G = nx.erdos_renyi_graph(num_vertices, edge_connectivity)

for i in range(5, 31, 5):
    
    start_trial = time.time()
    
    unvacc_rate = i/100
    
    G = load_graph(unvacc_rate=unvacc_rate)

    vertices = G.nodes
    print("Number of Vertices:", len(vertices))

    # List of edges
    edges = G.edges
    print("Number of Edges:", len(edges))

    # set the budget on the number of vertices that can be vaccinated
    budget = int(0.2*len(vertices))
    test_sample_size = 1000
    sample_size = 1000
    leak_probability = 0.2
    epsilon = 0.5

    # ---------------   Generate Samples    ------------------ #

    start_setup = time.time()

    # Generate samples for leaky vaccine on vertices (1 for successful vaccination, 0 for leak)
    samples = generate_samples(len(vertices), sample_size, leak_probability)
    print("Number of Samples:", len(samples))

    end_setup = time.time()
    print("Time to Setup LP:", end_setup-start_setup)

    # --------------------   Solve LP    --------------------- #

    lp_solution = get_lp_solution(vertices, edges, samples, budget, epsilon)
    vaccinated_vertices = lp_solution["rounded_solution"]

    # --------------------   Evaluate LP    --------------------- #

    start_eval = time.time()
    new_samples = generate_samples(len(vaccinated_vertices), test_sample_size, leak_probability)
    print("New Test Samples Setup")
    avg_degree_og = calculate_avg_degree(G)
    avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)
    print("Calculated Avg Degree")
    max_degree_og = calculate_max_degree(G)
    max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)
    print("Calculated Max Degree")
    #spectral_radius_obj = evaluate_spectral_radius(G, vaccinated_vertices, new_samples)
    #print("Calculated Spectral Radius")
    
    print("Original Average Degree:", avg_degree_og, "Simulated Average Degree:", avg_degree_obj)
    print("Original Max Degree:", max_degree_og, "Simulated Max Degree:", max_degree_obj)
    #print("Simulated Spectral Radius:", spectral_radius_obj)
    end_eval = time.time()
    print("Time to Evaluate Solution:", end_eval - start_eval)
    
    end_trial = time.time()

    filename = "avg_degree_mont_vertices_run.json"

    file_data = [{
                "num_vertices": len(G.nodes),
                "num_edges": len(G.edges),
                "sample_size": sample_size,
                "budget": budget,
                #"edge_connectivity": edge_connectivity,
                "leak_probability": leak_probability,
                 #"given_solution": lp_solution["given_solution"],
                "rounded_solution_size": len(lp_solution["rounded_solution"]),
                "lp_objective": lp_solution["lp_objective"],
                "lp_time": lp_solution["total_time"],
                "original_avg_degree": avg_degree_og,
                "original_max_degree": max_degree_og,
                "evaluated_avg_degree": avg_degree_obj,
                "evaluated_max_degree": max_degree_obj,
                #"evaluated_spectral_radius": spectral_radius_obj,
                "total_time": end_trial-start_trial
    }]

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            file_data = json.load(f) + file_data
    with open(filename, 'w') as f:
        json.dump(file_data, f)

    print(f"------------- {unvacc_rate} -------------------")