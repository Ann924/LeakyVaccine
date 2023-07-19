import time
import networkx as nx
import sys
import os
import json
from lp_utils import *
from data_utils import *
import random

#graph_seed = random.randint(0, 10**10)
graph_seed = 42
G = load_graph()
G = find_neighborhood(G, size=1000, seed=graph_seed)
#G = find_neighborhood(G, size=20)

vertices = G.nodes
print("Number of Vertices:", len(vertices))

# List of edges
edges = G.edges
print("Number of Edges:", len(edges))

# set the budget on the number of vertices that can be vaccinated
budget = int(0.2*len(vertices))
test_sample_size = 250
sample_size = 250
leak_probability = 0.2
epsilon = 0.5
filename = f"mont_leak_run_{len(vertices)}_narrow_{graph_seed}.json"
transmission_probability = 0.2
initial_infection_size = 15
infection_trials = 15

for t in range(15):
    
    if t<5:
        print(f"Skip Trial {t}")
        continue
    elif t==5:
        leak_lower = 250
    else:
        leak_lower = 25

    for leak in range(leak_lower, 310, 25):

        leak_probability = leak/1000

        start_trial = time.time()

        # ---------------   Generate Samples    ------------------ #

        start_setup = time.time()

        # Generate samples for leaky vaccine on vertices (1 for successful vaccination, 0 for leak)
        samples = generate_samples(len(vertices), sample_size, leak_probability)
        print("Number of Samples:", len(samples))

        end_setup = time.time()
        print("Time to Setup LP:", end_setup-start_setup)

        # --------------------   Solve Average Degree LP    --------------------- #

        lp, vertex_edge_dict = set_lp_constraints(vertices, edges, samples)
        lp = set_lp_budget(lp, budget)
        lp = set_lp_objective(lp, vertex_edge_dict, sample_size)
        lp_solution = get_lp_solution(lp, vertices, edges, samples, epsilon)
        vaccinated_vertices = lp_solution["rounded_solution"]

        # --------------------   Evaluate LP    --------------------- #

        start_eval = time.time()

        new_samples = generate_samples(len(vertices), test_sample_size, leak_probability)

        avg_degree_og = calculate_avg_degree(G)
        avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)

        max_degree_og = calculate_max_degree(G)
        max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)

        initial_infection_sets = generate_infection_sets(G, initial_infection_size, infection_trials)
        infection_size_og = evaluate_infection_spread(G, set(), new_samples, infection_set_trials = initial_infection_sets, transmission_probability=transmission_probability)
        infection_size_obj = evaluate_infection_spread(G, vaccinated_vertices, new_samples, infection_set_trials = initial_infection_sets, transmission_probability=transmission_probability)

        #spectral_radius_obj = evaluate_spectral_radius(G, vaccinated_vertices, new_samples)
        #print("Calculated Spectral Radius")

        print("Original Average Degree:", avg_degree_og, "Simulated Average Degree:", avg_degree_obj)
        print("Original Max Degree:", max_degree_og, "Simulated Max Degree:", max_degree_obj)
        print("Original Infection Spread:", infection_size_og, "Simulated Infection Spread:", infection_size_obj)
        #print("Simulated Spectral Radius:", spectral_radius_obj)

        end_eval = time.time()

        print("Time to Evaluate Solution:", end_eval - start_eval)

        end_trial = time.time()

        file_data = [{
                    "lp_type": "avg_degree",
                    "graph_seed": graph_seed,
                    "num_vertices": len(G.nodes),
                    "num_edges": len(G.edges),
                    "vertices": list(G.nodes),
                    "edges": list(G.edges),
                    "vaccinated_vertices": list(vaccinated_vertices),
                    "sample_size": sample_size,
                    "budget": budget,
                    "leak_probability": leak_probability,
                    "rounded_solution_size": len(lp_solution["rounded_solution"]),
                    "lp_objective": lp_solution["lp_objective"],
                    #"lp_time": lp_solution["total_time"],
                    "original_avg_degree": avg_degree_og,
                    "original_max_degree": max_degree_og,
                    "original_infection_spread": infection_size_og,
                    "evaluated_avg_degree": avg_degree_obj,
                    "evaluated_max_degree": max_degree_obj,
                    "evaluated_infection_spread": infection_size_obj,
                    "total_time": end_trial-start_trial
        }]

        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                file_data = json.load(f) + file_data
        with open(filename, 'w') as f:
            json.dump(file_data, f)

        # --------------------   Solve Max Degree LP    --------------------- #

        lp = reset_lp(lp)
        lp = set_lp_objective(lp, vertex_edge_dict, sample_size, objective="max_degree")
        lp_solution = get_lp_solution(lp, vertices, edges, samples, epsilon)
        vaccinated_vertices = lp_solution["rounded_solution"]

        # --------------------   Evaluate LP    --------------------- #

        start_eval = time.time()

        avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)
        max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)
        infection_size_obj = evaluate_infection_spread(G, vaccinated_vertices, new_samples, infection_set_trials = initial_infection_sets, transmission_probability=transmission_probability)

        print("Original Average Degree:", avg_degree_og, "Simulated Average Degree:", avg_degree_obj)
        print("Original Max Degree:", max_degree_og, "Simulated Max Degree:", max_degree_obj)
        print("Original Infection Spread:", infection_size_og, "Simulated Infection Spread:", infection_size_obj)

        end_eval = time.time()

        print("Time to Evaluate Solution:", end_eval - start_eval)

        end_trial = time.time()

        file_data = [{
                    "lp_type": "max_degree",
                    "graph_seed": graph_seed,
                    "num_vertices": len(G.nodes),
                    "num_edges": len(G.edges),
                    "vertices": list(G.nodes),
                    "edges": list(G.edges),
                    "vaccinated_vertices": list(vaccinated_vertices),
                    "sample_size": sample_size,
                    "budget": budget,
                    "leak_probability": leak_probability,
                    "rounded_solution_size": len(lp_solution["rounded_solution"]),
                    "lp_objective": lp_solution["lp_objective"],
                    "original_avg_degree": avg_degree_og,
                    "original_max_degree": max_degree_og,
                    "original_infection_spread": infection_size_og,
                    "evaluated_avg_degree": avg_degree_obj,
                    "evaluated_max_degree": max_degree_obj,
                    "evaluated_infection_spread": infection_size_obj,
                    "total_time": end_trial-start_trial
        }]

        if os.path.isfile(filename):
            with open(filename, 'r') as f:
                file_data = json.load(f) + file_data
        with open(filename, 'w') as f:
            json.dump(file_data, f)

        print(f"------------- {graph_seed} --- {leak_probability} --- {t} -------------------")