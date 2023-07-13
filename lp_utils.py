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

def set_lp_constraints(vertices, edges, samples):
    try:
        start_lp_setup = time.time()
        
        # Create a new model
        m = gp.Model()

        # Create vertex-selection variables
        x = m.addVars(len(vertices), lb=[0]*len(vertices), ub=[1]*len(vertices))
        print("Created X Variables")

        # Create edge-coverage indicator variables
        # 0 = not covered, 1 = covered
        y = m.addMVar((len(samples), len(edges)), lb=0, ub=1)
        #y = m.addMVar(len(edges), lb=0, ub=len(samples))
        print("Created Y Variables")

        # Add budget constraint
        # m.addConstr(x.sum() <= budget, "budget_constraint")
        # print("Set Budget Constraint")

        # Add coverage constraint of edges
        vertex_edge_dict = {v:set() for v in range(len(vertices))}
        for i, edge in enumerate(edges):

            v1 = edge[0]
            v2 = edge[1]
            
            vertex_edge_dict[v1].add(i)
            vertex_edge_dict[v2].add(i)
            
            #m.addConstr(gp.quicksum(sample[v1]*x[v1] + sample[v2]*x[v2] for sample in samples) >= y[i])
            
            for j in range(len(samples)):
                m.addConstr(samples[j][v1]*x[v1] + samples[j][v2]*x[v2] >= y[j][i])
        print("Set Coverage Constraint")
        
        end_lp_setup = time.time()
        print("Total Setup Time:", end_lp_setup-start_lp_setup)
        
        return (m, x, y), vertex_edge_dict
    
    except gp.GurobiError as e:
        print('Error code ' + str(e.errno) + ': ' + str(e))

    except AttributeError:
        print('Encountered an attribute error')

        
def set_lp_budget(lp, budget):
    
    m, x, y = lp
    
    # Add budget constraint
    m.addConstr(x.sum() <= budget, "budget_constraint")
    print(f"Set Budget Constraint to {budget}")
    
    m.update()
    return m, x, y
    
def set_lp_objective(lp, vertex_edge_dict, sample_size, objective="avg_degree"):
    
    m, x, y = lp
    
    if objective=="avg_degree":
        # Maximize the number of edges covered
        m.setObjective(y.sum(), GRB.MAXIMIZE)
    elif objective=="max_degree":
        num_vertices = len(vertex_edge_dict.keys())
        z = m.addMVar((sample_size, num_vertices), lb=0, ub=len(vertex_edge_dict.keys()))
        z_max = m.addVars(sample_size, lb=[0]*sample_size, ub=[len(vertex_edge_dict.keys())]*sample_size)
        
        for sample in range(sample_size):
            for i in range(num_vertices):
                m.addConstr(len(vertex_edge_dict[i]) - gp.quicksum(y[sample][e] for e in vertex_edge_dict[i]) == z[sample][i])
            # Calculate maximum degree per sample
            m.addConstr(z_max[sample] == gp.max_(z[sample].tolist()))
        
        # Minimize the maximum number of uncovered edges
        m.setObjective(z_max.sum(), GRB.MINIMIZE)
        
    m.update()
    return m, x, y

def reset_lp(lp, keep_budget=True):
    m, x, y = lp
    m.reset()
    
    if not keep_budget:
        constr = m.getConstrByName("budget_constraint")
        if constr:
            m.remove(constr)
    
    return (m, x, y)

def get_lp_solution(lp, vertices, edges, samples, epsilon):
    m, x, y = lp
    
    start_lp_solution = time.time()
        
    # Optimize model
    m.optimize()
    print('Obj: %g' % m.ObjVal)

    x_rounded = lp_round(epsilon, x)
    print("Rounded Solution Size:", len(x_rounded))

    end_lp_solution = time.time()
    print("Time to Solve LP:", end_lp_solution-start_lp_solution)

    return {"given_solution": list([float(x[i].X) for i in range(len(x.keys()))]),
            "rounded_solution": list(x_rounded),
            "lp_objective": m.ObjVal}
            #"lp_objective": ((len(edges) - m.ObjVal/len(maxamples))/len(vertices))}
    
    #except gp.GurobiError as e:
    #    print('Error code ' + str(e.errno) + ': ' + str(e))

    #except AttributeError:
    #    print('Encountered an attribute error')

# TODO: Round differently for max v. avg degree LP
def lp_round(epsilon, x):
    
    lmbda = 2*(1-epsilon)
    
    cover = set()
    for i, x_var in x.items():
        if (x_var.X) >= 1/lmbda:
            cover.add(i)
        else:
            # round to 1 with probability lmbda * x
            if random.random() <= lmbda*(x_var.X):
                cover.add(i)
    return cover

def calculate_avg_degree(G):
    return 2*len(G.edges)/len(G.nodes)

def calculate_max_degree(G):
    return max(G.degree(), key=lambda x:x[1])[1]

"""
Returns the average degree of the resulting vaccination graph
"""
def evaluate_avg_degree(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = set(vaccinated_vertices)
    vertices = np.array(list(G.nodes))
    
    total_edges = len(G.edges)
    removed_edges = 0
    for s in samples:
        successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
        removed_edges += len(G.edges(successful_vaccinations))
    return 2*(total_edges - (removed_edges/len(samples)))/len(G.nodes)

"""
Returns the max degree of the resulting vaccination graph
"""
def evaluate_max_degree(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = set(vaccinated_vertices)
    vertices = np.array(list(G.nodes))
    
    total_max = 0
    for s in samples:
        
        G_copy = G.copy()
        successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
        removed_edges = G.edges(successful_vaccinations)
        G_copy.remove_edges_from(removed_edges)
        
        max_degree = max(G_copy.degree(), key=lambda x:x[1])[1]
        total_max += max_degree
        
    return total_max/len(samples)

def generate_infection_sets(G, infection_set_size, number_of_sets):
    
    infection_set_list = []
    for trial in range(number_of_sets):
        infected_set = set([])
        while len(infected_set)<infection_set_size:
            chosen = random.randint(0, len(G.nodes)-1)
            infected_set.add(chosen)
        infection_set_list.append(infected_set)

    return infection_set_list

def evaluate_infection_spread(G, vaccinated_vertices, samples, infection_set_trials = [set([])], transmission_probability=1, trials=10):
    
    vaccinated_vertices = set(vaccinated_vertices)
    vertices = np.array(list(G.nodes))
    
    infected_size_list = []
    
    for s in samples:
        
        if len(vaccinated_vertices)>0:
            G_copy = G.copy()
            successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
            removed_edges = G.edges(successful_vaccinations)
            G_copy.remove_edges_from(removed_edges)
        else:
            G_copy = G
        
        for i in range(trials):
            
            for initial_infection in infection_set_trials:
                
                infected = set(initial_infection)
                visited = set(initial_infection)
                queue = list(initial_infection)
                
                while len(queue)>0:

                    infected_v = queue.pop(0)

                    if random.random() <= transmission_probability:
                        infected.add(infected_v)

                        for edge in G_copy.edges([infected_v]):
                            
                            v1, v2 = edge
                            
                            if v1 not in visited:
                                queue.append(v1)
                            if v2 not in visited:
                                queue.append(v2)

                            visited.add(v1)
                            visited.add(v2)
            
                infected_size_list.append(len(infected))
    
    return sum(infected_size_list)/len(infected_size_list)
        
"""
Returns the spectral radius of the resulting vaccination graph
"""
def evaluate_spectral_radius(G, vaccinated_vertices, samples):
    # samples contains indicator variables for whether the vertices leak the disease after being vaccinated
    
    vaccinated_vertices = np.array(list(vaccinated_vertices))
    vertices = list(G.nodes)
    
    total_spectral_radius = 0
    for s in samples:
        G_copy = G.copy()
        successful_vaccinations = [v for v in vertices[s==1] if v in vaccinated_vertices]
        removed_edges = G.edges(successful_vaccinations)
        G_copy.remove_edges_from(removed_edges)
        spectral_radius = max(nx.adjacency_spectrum(G_copy))
        total_spectral_radius += spectral_radius.real
    return total_spectral_radius/len(samples)
# ---------------   Define variables    ------------------ #

# B is the budget on the number of vertices that can be vaccinated
'''budget = 5
num_vertices = 50
edge_connectivity = 0.1
sample_size = 100
leak_probability = 0.2
epsilon = 0.5
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

start_lp = time.time()
lp, vertex_edge_dict = set_lp_constraints(vertices, edges, samples, budget)
lp = set_lp_objective(lp, vertex_edge_dict)
lp_solution = get_lp_solution(lp, vertices, edges, samples, epsilon)
end_lp = time.time()

vaccinated_vertices = lp_solution["rounded_solution"]

print("Total LP Time:", end_lp - start_lp)

# --------------------   Evaluate LP    --------------------- #

new_samples = generate_samples(len(vaccinated_vertices), test_sample_size, leak_probability)
avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)
max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)
#spectral_radius_obj = evaluate_spectral_radius(G, vaccinated_vertices, new_samples)
print("Simulated Average Degree:", avg_degree_obj)
print("Simulated Max Degree:", max_degree_obj)
#print("Simulated Spectral Radius:", spectral_radius_obj)

avg_degree_run = {"vertices": list(vertices),
               "edges": list(edges),
               "budget": budget,
               "given_solution": lp_solution["given_solution"],
               "rounded_solution": lp_solution["rounded_solution"],
               "lp_objective": lp_solution["lp_objective"],
               "evaluated_avg_degree_objective": avg_degree_obj,
               "evaluated_max_degree_objective": max_degree_obj
              }

start_lp = time.time()
lp = reset_lp(lp)
lp = set_lp_objective(lp, vertex_edge_dict, sample_size, objective="max_degree")
lp_solution = get_lp_solution(lp, vertices, edges, samples, epsilon)
end_lp = time.time()

vaccinated_vertices = lp_solution["rounded_solution"]

print("Total LP Time:", end_lp - start_lp)

new_samples = generate_samples(len(vaccinated_vertices), test_sample_size, leak_probability)
avg_degree_obj = evaluate_avg_degree(G, vaccinated_vertices, new_samples)
max_degree_obj = evaluate_max_degree(G, vaccinated_vertices, new_samples)
spectral_radius_obj = evaluate_spectral_radius(G, vaccinated_vertices, new_samples)
print("Simulated Average Degree:", avg_degree_obj)
print("Simulated Max Degree:", max_degree_obj)
print("Simulated Spectral Radius:", spectral_radius_obj)

max_degree_run = {"vertices": list(vertices),
               "edges": list(edges),
               "budget": budget,
               "given_solution": lp_solution["given_solution"],
               "rounded_solution": lp_solution["rounded_solution"],
               "lp_objective": lp_solution["lp_objective"],
               "evaluated_avg_degree_objective": avg_degree_obj,
               "evaluated_max_degree_objective": max_degree_obj
              }

#with open("avg_max_degree_run_test.json", 'w') as f:
#   json.dump([avg_degree_run, max_degree_run], f)'''