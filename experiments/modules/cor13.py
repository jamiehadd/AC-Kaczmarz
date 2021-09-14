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
