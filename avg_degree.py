import sys
import gurobipy as gp
from gurobipy import GRB
import random
import numpy as np
import networkx as nx
import time
import json

def generate_samples(number_of_vertices, number_of_samples, leak_probability):
    samples = np.zeros((number_of_samples, number_of_vertices))
    for i in range(number_of_samples):
        samples[i, :] = np.random.choice([0, 1], size=number_of_vertices, p=[leak_probability, 1-leak_probability])
    return samples

def get_lp_solution(vertices, edges, samples, budget, epsilon):
    try:
        start_lp_setup = time.time()
        
        # Create a new model
        m = gp.Model()

        # Create vertex-selection variables
        x = m.addMVar(len(vertices), lb=0, ub=1)
        print("Created X Variables")

        # Create edge-coverage indicator variables
        # 0 = not covered, 1 = covered
        y = m.addMVar((len(samples), len(edges)), lb=0, ub=1)
        print("Created Y Variables")

        # Set objective
        m.setObjective(y.sum(), GRB.MAXIMIZE)

        # Add budget constraint
        m.addConstr(x.sum() <= budget)
        print("Set Budget Constraint")

        # Add coverage constraint of edges
        for i, edge in enumerate(edges):

            v1 = edge[0]
            v2 = edge[1]

            for j in range(len(samples)):
                m.addConstr(samples[j][v1]*x[v1] + samples[j][v2]*x[v2] >= y[j][i])
        print("Set Coverage Constraint")

        end_lp_setup = time.time()
        print("Time to Setup LP:", end_lp_setup-start_lp_setup)

        start_lp_solution = time.time()
        
        # Optimize model
        m.optimize()
        print('Obj: %g' % m.ObjVal)
        print('Average Obj: %g' % (m.ObjVal/len(samples)))
        print('Average Remaining Edges Obj: %g' % (len(edges) - m.ObjVal/len(samples)))
        print('Average Degree Obj: %g' % ((len(edges) - m.ObjVal/len(samples))/len(vertices)))

        x_rounded = lp_round(epsilon, x)
        print("Rounded Solution Size:", len(x_rounded))

        end_lp_solution = time.time()
        print("Time to Solve LP:", end_lp_solution-start_lp_solution)
        
        return {"given_solution": list([float(var.X) for var in x]),
                "rounded_solution": list(x_rounded),
                "lp_objective": ((len(edges) - m.ObjVal/len(samples))/len(vertices)),
                "total_time": end_lp_solution - start_lp_setup}
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

def lp_round(epsilon, x):
    
    lmbda = 2*(1-epsilon)
    
    cover = set()
    for i, x_var in enumerate(x):
        if (x_var.X) >= 1/lmbda:
            cover.add(i)
        else:
            # round to 1 with probability lmbda * x
            if random.random() <= lmbda*(x_var.X):
                cover.add(i)
    return cover

"""
Returns the average degree of the resulting vaccination graph
"""
def evaluate_vaccination(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = np.array(list(vaccinated_vertices))
    
    total_edges = len(G.edges)
    removed_edges = 0
    for s in samples:
        successful_vaccinations = vaccinated_vertices[s==1]
        removed_edges += len(G.edges(successful_vaccinations))
    return (total_edges - (removed_edges/len(samples)))/len(G.nodes)
    
# ---------------   Define variables    ------------------ #

# B is the budget on the number of vertices that can be vaccinated
'''budget = 10
num_vertices = 50
edge_connectivity = 0.5
sample_size = 100
leak_probability = 0.2
epsilon = 0.01
test_sample_size = 1000

num_vertices = int(sys.argv[1]) if len(sys.argv)>1 else num_vertices
edge_connectivity = float(sys.argv[2]) if len(sys.argv)>2 else edge_connectivity
sample_size = int(sys.argv[3]) if len(sys.argv)>3 else sample_size
leak_probability = float(sys.argv[4]) if len(sys.argv)>4 else leak_probability

# ---------------   Generate Samples    ------------------ #

start_setup = time.time()

# Generate Erdos Renyi graph
G = nx.erdos_renyi_graph(num_vertices, edge_connectivity)

vertices = G.nodes
print("Number of Vertices:", len(vertices))

# List of edges
edges = G.edges
print("Number of Edges:", len(edges))

# Generate samples for leaky vaccine on vertices (1 for successful vaccination, 0 for leak)
samples = generate_samples(len(vertices), sample_size, leak_probability)
print("Number of Samples:", len(samples))

end_setup = time.time()
print("Time to Setup Samples:", end_setup-start_setup)

# --------------------   Solve LP    --------------------- #

lp_solution = get_lp_solution(vertices, edges, samples, budget, epsilon)
vaccinated_vertices = lp_solution["rounded_solution"]

# --------------------   Evaluate LP    --------------------- #

new_samples = generate_samples(len(vaccinated_vertices), test_sample_size, leak_probability)
avg_degree_obj = evaluate_vaccination(G, vaccinated_vertices, new_samples)
print("Simulated Average Degree:", avg_degree_obj)

with open("avg_degree_run.json", 'w') as f:
    
    json.dump({"vertices": list(vertices),
               "edges": list(edges),
               "budget": budget,
               "given_solution": lp_solution["given_solution"],
               "rounded_solution": lp_solution["rounded_solution"],
               "lp_objective": lp_solution["lp_objective"],
               "lp_time": lp_solution["total_time"],
               "evaluated_objective": avg_degree_obj
              }, f)'''