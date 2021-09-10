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
import helper_functions
import cor13

# Wraps helper functions and applies them to specific conditions

# Independent Edge Sets
# Provides a non-overlapping independent edge cover over a graph G
# Returns a list of list of edges
def independent_edge_sets(G):
    # use maximal_matching!
    # Take a copy of G
    H = G.copy()
    ies = []
    while len(H.edges())> 0:
        ieset = maximal_matching(H)
        ies.append(ieset)
        H.remove_edges_from(ieset)
    return ies

def blocks_from_ies(G, A):
    subs = independent_edge_sets(G)
    blks = blocks_edge(A, subs)
    return blks

# cliques
# Grabs subgraphs that correspond to cliques within a graph
# Input: Incidence matrix, graph

def blocks_from_cliques(G, A):
    cliques = list(find_cliques(G)) # find_cliques grab maximal cliques for each node
    clique_edge = []
    for clique in cliques:
        clique_edge.append(edges_from_pnts(clique))
    blks = blocks_edge(A, clique_edge)
    blks = remove_trivial_subgraphs(blks)
    return blks

def largest_clique(cliques):
    list_len = [len(i) for i in cliques]
    indice = np.argmax(np.array(list_len))
    return cliques[indice]

def largest_clique_bounded(cliques, bound):
    list_len = [len(i) for i in cliques]
    if bound != None:
        for i in range(len(list_len)):
            if list_len[i] > bound:
                list_len[i] = 0
    indice = np.argmax(np.array(list_len))
    return cliques[indice]

# This is an EDGE covering with an upper bound
def clique_edge_cover(G, A, bound=None):
    H = G.copy()
    cliques_list = []
    while len(H.edges)>0:
        foo = largest_clique_bounded(list(find_cliques(H)), bound)
        H.remove_nodes_from(foo)
        foo = edges_from_pnts(foo)
        cliques_list.append(foo)
        # H.remove_edges_from(foo)
        # remove nodes or edges? remove edges only!
    cliques_list = blocks_edge(A, cliques_list)
    return cliques_list

# This is a NODE covering with an upper bound on clique size
def clique_node_cover(G, A, bound=None):
    H = G.copy()
    cliques_list = []
    while len(H.nodes)>0:
        foo = largest_clique_bounded(list(find_cliques(H)), bound)
        H.remove_nodes_from(foo) # Removes nodes because we want a node covering
        foo = edges_from_pnts(foo)
        cliques_list.append(foo)
    cliques_list = blocks_edge(A, cliques_list)
    return cliques_list


# Paths
# Path Gossip has its OWN block RK algorithm, shown below

# helper functions for path selection
# path with no restrictions except length
# if a path terminates before the length, the short path is returned
def find_path(G, r, l):
    path = [r]
    path.append(find_edge(G, r, -1))
    while len(path)<l:
        path.append(find_edge(G, path[len(path)-1], path[len(path)-2]))
    return path

def find_edge(G, r, e=-1):
    # e = previous node (so we don't choose consecutive repeating edges)
    if e != -1: # no previous node
        neighbors = [n for n in G.neighbors(r)]
        if len(neighbors)==0:
            print("no neighbors found, isolated point")
            return
        else:
            a = neighbors[np.random.randint(0, len(neighbors)-1)]
            return a
    else:
        neighbors = [n for n in G.neighbors(r)]
        neighbors = list(filter(lambda x: x!=e, neighbors))
        if len(neighbors)<=1:
            print("no neighbors found, path terminates here")
            return
        else:
            a = neighbors[np.random.randint(0, len(neighbors)-1)]
            return a

def path_blk(A, G, r, l):
    path = find_path(G, r, l)
    blk = find_subgraph_from_edges(A, edges_from_pnts(path))
    return blk

# block RK for paths
# since paths are selected randomly at each step, returns an additional paths which is the list of blks chosen
def blockRK_path(A, G, sol, b, N, c, l):
    x = c
    x_list = [x]
    errors = [np.linalg.norm(x-sol)]
    paths = []
    for j in range(1, N+1):
        r = randint(len(G.nodes))
        blk = path_blk(A, G, r, l)
        paths.append(blk)
        x = x + np.linalg.pinv(A[blk,:])@(b[blk] - A[blk,:]@x)
        errors.append(np.linalg.norm(x-sol))
        x_list.append(np.asarray(x))
    return paths, x, x_list, errors
