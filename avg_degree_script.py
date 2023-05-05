import time
import networkx as nx
import sys
import os
import json
from avg_degree import generate_samples, get_lp_solution, evaluate_vaccination

# B is the budget on the number of vertices that can be vaccinated
budget = 10
num_vertices = 50
edge_connectivity = 0.5
sample_size = 100
leak_probability = 0.2
epsilon = 0.1

for trial in range(10):
    G = nx.erdos_renyi_graph(num_vertices, edge_connectivity)
    
    vertices = G.nodes
    print("Number of Vertices:", len(vertices))

    # List of edges
    edges = G.edges
    print("Number of Edges:", len(edges))
    
    for sample_size in range(100, 1100, 100):

        # ---------------   Generate Samples    ------------------ #

        start_setup = time.time()

        # Generate samples for leaky vaccine on vertices (1 for successful vaccination, 0 for leak)
        samples = generate_samples(len(vertices), sample_size, leak_probability)
        print("Number of Samples:", len(samples))

        end_setup = time.time()
        print("Time to Setup Samples:", end_setup-start_setup)

        # --------------------   Solve LP    --------------------- #

        lp_solution = get_lp_solution(vertices, edges, samples, budget)
        vaccinated_vertices = lp_solution["rounded_solution"]

        # --------------------   Evaluate LP    --------------------- #

        new_samples = generate_samples(len(vaccinated_vertices), sample_size, leak_probability)
        avg_degree_obj = evaluate_vaccination(G, vaccinated_vertices, new_samples)
        print("Simulated Average Degree:", avg_degree_obj)

        file_data = [{"vertices": list(vertices),
                     "edges": list(edges),
                     "sample_size": sample_size,
                     "budget": budget,
                     "edge_connectivity": edge_connectivity,
                     "leak_probability": leak_probability,
                     "given_solution": lp_solution["given_solution"],
                     "rounded_solution": lp_solution["rounded_solution"],
                     "lp_objective": lp_solution["lp_objective"],
                     "lp_time": lp_solution["total_time"],
                     "evaluated_objective": avg_degree_obj}]

        if os.path.isfile("avg_degree_samplesize_runs.json"):
            with open("avg_degree_samplesize_runs.json", 'r') as f:
                file_data = json.load(f) + file_data
        with open("avg_degree_samplesize_runs.json", 'w') as f:
            json.dump(file_data, f)
        
        print(f"------------- SAMPLES SIZE {sample_size}: TRIAL {trial} -------------------")