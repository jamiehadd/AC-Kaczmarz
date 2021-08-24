# Import Libraries
import matplotlib.pyplot as plt
import networkx as nx
import itertools
import numpy as np
import scipy as sp
from numpy.linalg import pinv
from numpy.random import randint, normal
from networkx.algorithms.clique import find_cliques, enumerate_all_cliques, make_max_clique_graph, graph_number_of_cliques
from networkx.algorithms.matching import maximal_matching
from networkx.algorithms.operators.unary import complement
from networkx.linalg.algebraicconnectivity import algebraic_connectivity
from scipy import stats
import itertools as it
from collections import Counter


# Helper functions for grabbing row indices for subgraphs

# Converts sets of edges into row indices in the incidence matrix
def find_subgraph_from_edges(A,edges):
    edge_indices = []
    for edge in edges:
        for i in range(A.shape[0]):
            if A[i,edge[0]] != 0 and A[i,edge[1]] != 0:
                edge_indices.append(i)
    if len(edges) != len(edge_indices):
        print("Did not find all edges of subgraph in incidence matrix.")
    return edge_indices

# Given a list of points, returns a list of corresponding rows in incidence matrix

def blocks_pnts(A, subgraphs):
    blocks = []
    for subgraph in subgraphs:
        blocks.append(find_subgraph_from_pnts(A, subgraph))
    return blocks

# Given a list of edges, returns a list of corresponding rows in incidence matrix
def blocks_edge(A, subgraphs):
    blocks = []
    for subgraph in subgraphs:
        blocks.append(find_subgraph_from_edges(A, subgraph))
    return blocks

# Turns a list of points into a list of edges
def edges_from_pnts(subgraph):
    edges = []
    n = len(subgraph)
    for i in range(n-1):
        edges.append((subgraph[i], subgraph[i+1]))
    return edges
