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

# Corrupted block RK with one-off random additive noise (Gaussian)
# Adds a Gaussian noise vector to b at first

def blockRK_AGN(A, sol, b, blocks, N, c, s=0.05, m=0):
    k = len(blocks)
    x = c
    errors = []
    x_list = [x]
    err = normal(m, s, b.shape)
    b = b + err
    for j in range (1, N+1):
        x = x + np.linalg.pinv(A[blocks[i],:])@(b[blocks[i]] - A[blocks[i],:]@x) + err
    errors.append(np.linalg.norm(x-sol))
    x_list.append(x)
    return x, x_list, errors

# Corrupted block RK with two modes of failure: ONE-OFF additive gaussian noise and
# random link failure

def blockRK_cor(A, sol, b, blocks, N, c, p=0.3, m=0, s=0.1):
    k = len(blocks)
    x = c
    errors = []
    x_list = []
    err = normal(m, s, b.shape)
    b = b + err
    for j in range (1, N+1):
        r = stats.bernoulli.rvs(p, size = 1);
        i = randint(k)
        if r[0] == 1:
            x = x + np.linalg.pinv(A[blocks[i],:])@(b[blocks[i]] - A[blocks[i],:]@x)
            errors.append(np.linalg.norm(x-sol))
    return x, x_list, errors
