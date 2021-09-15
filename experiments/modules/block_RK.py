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
from networkx.generators.random_graphs import erdos_renyi_graph
from networkx.linalg.algebraicconnectivity import algebraic_connectivity
from scipy import stats
import itertools as it
from collections import Counter
from networkx.generators.lattice import grid_graph
from more_itertools import locate
from ipywidgets import IntProgress
from IPython.display import display
import time



# Standard Block RK
# x_list allows us to track evolution of x and its individual components (for "collapse" graph?)
# Takes in c: initial vector, b: Ax=b (column of zeros), A: incidence matrix, and sol: solution to Ax=b
# Also takes in list of lists (Blocks) where each block is a list of rows within the incidence matrix corresponding to
# specific subgraphs and N, number of maximum iterations
# Returns: final value of x, list of x over iterations, and error over iterations

def blockRK(A, sol, b, blocks, N, c):
    k = len(blocks)#Block-RK-Functions
    x = c
    x_list = [x]
    errors = []
    for j in range (1, N+1):
        i = randint(k);
        x = x + np.linalg.pinv(A[blocks[i],:])@(b[blocks[i]] - A[blocks[i],:]@x)
        errors.append(np.linalg.norm(x-sol))
        x_list.append(np.asarray(x))
    return x, x_list, errors

# Corrupted block RK with random link failure (RLF)
# Fails on each block randomly from a normal distribution
# Independent but not identical probability of failure
# Accepts fixed failure probability (Bernoulli)

def blockRK_RLF(A, sol, b, blocks, N, c, p=0.3):
    k = len(blocks)
    x = c
    x_list = [x]
    errors = []
    for j in range (1, N+1):
        r = stats.bernoulli.rvs(p, size = 1);
        i = randint(k)
        if r[0] == 1:
            x = x + np.linalg.pinv(A[blocks[i],:])@(b[blocks[i]] - A[blocks[i],:]@x)
        errors.append(np.linalg.norm(x-sol))
        x_list.append(x)
    return x, x_list, errors

# Corrupted block RK with Constant Edge Communication Error
# Where a constant error vector is added to b according to some specified probability (bernoulli dist.)

def blockRK_cece(A, sol, b, blocks, N, c, err, p=0.2): # where err is a constant vector
    k = len(blocks)
    x = c
    x_list = [x]
    errors = []
    m_t = b
    for j in range(1, N+1):
        r = stats.bernoulli.rvs(p, size = 1)
        i = randint(k)
        if r[0] == 1:
            m_t = b + err
        else:
            m_t = b
        x = x + np.linalg.pinv(A[blocks[i],:])@(b[blocks[i]] - A[blocks[i],:]@x)
        errors.append(np.linalg.norm(x-sol))
        x_list.append(x)
    return x, x_list, errors

# Corrupted block RK with Varying Edge Communication Error

def blockRK_vece(A, sol, b, blocks, N, c, err, p=0.2): # where err is a distribution of possible error values
    k = len(blocks)
    x = c
    x_list = [x]
    errors = []
    m_t = b
    for j in range(1, N+1):
        r = stats.bernoulli.rvs(p, size = 1)
        i = randint(k)
        if r[0] == 1:
            y = err.shape[1]
            t = randint(y)
            m_t = b + err[t]
        else:
            m_t = b
        x = x + np.linalg.pinv(A[blocks[i],:])@(b[blocks[i]] - A[blocks[i],:]@x)
        errors.append(np.linalg.norm(x-sol))
        x_list.append(x)
    return x, x_list, errors

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

# Functions to find the appropriate rate constant in Corollary 1.3

def eigenvalue(blk, A):
    blks = [A[i] for i in blk]
    mat = np.concatenate(blks, axis=0)
    eigs, vec = np.linalg.eig(mat*mat.transpose())
    # We want minimum non-zero eigenvalue and maximum non-zero eigenvalue
    mineig = np.min(eigs[np.nonzero(eigs)])
    maxeig = np.max(eigs[np.nonzero(eigs)])
    return mineig, maxeig

def alpha(blks,A):
    min_foo = [eigenvalue(blk, A)[0] for blk in blks]
    max_foo = [eigenvalue(blk, A)[1] for blk in blks]
    alpha = np.min(min_foo)
    beta = np.max(max_foo)
    return alpha, beta

def rR(blks):
    #Counter(it.chain.from_iterable(map(set, blks))
# OR
    counts = Counter(x for xs in blks for x in set(xs))
    R = counts.most_common()[0][1]
    r = counts.most_common()[-1][1]
    return r, R

def findM(blks):
    M = np.max([len(blk) for blk in blks])
    return M

# Corollary 1.3

# Independent edge sets
def rate_ies(G, blks):
    k = len(blks)
    bound = 1-rR(blks)[0]*algebraic_connectivity(G)/(2*k)
    return bound

# Paths
def rate_paths(G, blks):
    k = len(blks)
    bound = 1-rR(blks)[0]*algebraic_connectivity(G)/(4*k)
    return bound

# Cliques
def rate_cliques(G, blks):
    k = len(blks)
    bound = 1-rR(blks)[0]*algebraic_connectivity(G)/(4*k)
    return bound

# Arbitrary Blocks
def rate_arbi(G, blks):
    k = len(blks)
    M = findM(blks)
    bound = 1-rR(blks)[0]*algebraic_connectivity(G)/(M*k)
    return bound

def sigma(errs):
    # turns list of lists into array retrieves covariance matrix
    mat = np.array(errs)
    sig = np.cov(errs)
    return sig

# Two main graphs: Collapse Plot (Individual node values over iteration)
# and Error Plot (Error over iteration with appropriate bound)

# n is number of nodes in graph!
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
def collapse_plt(x_list, n, N):
    x_axis = x_list.copy()
    for i in range(N+1):
        x_axis[i] = np.concatenate(x_axis[i])
        # np.append(x_axis[i], x_bar)
    for i in range (n):
        plt.plot(range(N+1), [x_axis[f][i] for f in range(N+1)], linewidth=3)
    plt.xlabel('Iteration number, $k$', fontsize=15)
    plt.ylabel('Node values, $x_i$', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

def error_plt(errors, G, blks, sol, N, rate='arbi'):
    if rate == 'cliques':
        r = rate_cliques(G, blks)
        label = 'Block Gossip (Cliques)'
    elif rate == 'ies':
        r = rate_ies(G,blks)
        label = 'Block Gossip (Independent Edge Sets)'
    elif rate == 'path':
        r = rate_paths(G, blks)
        label = 'Block Gossip (Paths)'
    elif rate == 'arbi':
        r = rate_arbi(G, blks)
        label = 'Block Gossip'
    else:
        print('rate not supported, using arbitrary blk rate')
        r = rate_arbi(G, blks)
    blabel = r'Predicted Bound'
    bound = [(r**i)*(errors[0]**2) for i in range(N)]
    err = [errors[i]**2 for i in range(len(errors))]
    plt.semilogy(range(N),err[0:N], 'b', linewidth=4, label = label)
    plt.semilogy(range(N), bound, 'r--', linewidth=4, label = blabel)
    plt.legend(prop={'size': 15})
    plt.xlabel('Iteration number, $k$', fontsize=15)
    plt.ylabel(r'$||c_k-c*||^2$', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return bound, r, err

def error_plt_ece(errors, A, G, blks, sol, N, err, t):
    k = len(blks)
    alp = alpha(blks, A)[0]
    beta = alpha(blks, A)[1]
    r = rR(blks)[0]
    R = rR(blks)[1]
    rate = 1-(r*algebraic_connectivity(G)/(beta*k))
    if t == 'cece' or 'CECE' or 'constant':
        mc = beta*R/(alp*r*algebraic_connectivity(G))*np.linalg.norm(err)
        label = r'Block Gossip (CECE)'
    elif t == 'vece' or 'VECE' or 'random' or 'varying':
        mc = beta*R/(alp*r*algebraic_connectivity(G))*np.trace(sigma(err))
        label = r'Block Gossip (VECE)'
    else:
        mc = 0
    bound = [(rate**i)*(errors[0]**2)+mc for i in range(N+1)]
    err = [errors[i]**2 for i in range(len(errors))]
    plt.semilogy(range(np.shape(errors)[0]),err, 'b', linewidth=4, label = label)
    plt.semilogy(range(np.shape(bound)[0]), bound, 'r--', linewidth=4, label = r'Predicted Bound')
    plt.legend(prop={'size': 15})
    plt.xlabel('Iteration number, $k$', fontsize=15)
    plt.ylabel(r'$||c_k-c*||^2$', fontsize=15)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)
    return bound

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
    max_count = 5000
    f = IntProgress(min=0, max=max_count)
    display(f)
    while len(H.edges)>0:
        foo = largest_clique_bounded(list(find_cliques(H)), bound)
        foo = edges_from_pnts(foo)
        H.remove_edges_from(foo)
        cliques_list.append(foo)
        f.value += 1
    f.value = max_count # signals end of run
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
def find_edge(G, r):
    neighbors = [n for n in G.neighbors(r)]
    if len(neighbors)==1:
        print("no more neighbors found, terminating")
        return
    else:
        a = neighbors[np.random.randint(0, len(neighbors))]
        return a

def find_path(G, r, l):
    path = [r]
    while len(path)<l:
        nb = find_edge(G, path[len(path)-1])
        while nb in path:
            nb = find_edge(G, path[len(path)-1])
        if nb == 'null':
            return path
        path.append(nb)
    return path

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
    max_count = N+1
    f = IntProgress(min=0, max=max_count)
    display(f)
    for j in range(1, N+1):
        r = randint(len(G.nodes))
        blk = path_blk(A, G, r, l)
        paths.append(blk)
        x = x + np.linalg.pinv(A[blk,:])@(b[blk] - A[blk,:]@x)
        errors.append(np.linalg.norm(x-sol))
        x_list.append(np.asarray(x))
        f.value += 1
    f.value = max_count
    return paths, x, x_list, errors

def random_blocks(A, s, bn):
    n = A.shape[0]
    blocks = []
    i = 0
    while i in range(bn):
        random_indices = np.random.choice(n, size=s, replace=False)
        blocks.append(list(random_indices))
        i = i + 1
    return blocks
