import networkx as nx
import random 

def load_graph(seed=42, unvacc_rate=1):
    
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
    
    if unvacc_rate < 1:
        nodes = list(G.nodes)
        for n in nodes:
            # simulate already vaccinated nodes
            if random.random()>unvacc_rate:
                G.remove_node(n)

        mapping = dict(zip(G, range(len(G.nodes))))
        G = nx.relabel_nodes(G, mapping) 

    return G

def find_neighborhood(G, size = 500, seed=42):
    
    random.seed(seed)
    
    neighborhood_vertices = set()
    starting_vertex = random.choice(list(G.nodes))
    queue = [starting_vertex]
    
    while len(neighborhood_vertices)<size:
        v = queue.pop(0)
        neighborhood_vertices.add(v)
        for v1, v2 in G.edges([v]):
            queue.append(v2)
    
    G = G.subgraph(list(neighborhood_vertices))
    mapping = dict(zip(G, range(len(G.nodes))))
    G = nx.relabel_nodes(G, mapping)
    
    return G