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

# Two main graphs: Collapse Plot (Individual node values over iteration)
# and Error Plot (Error over iteration with appropriate bound)

# n is number of nodes in graph!
# https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html
def collapse_plt(x_list, n, N, x_bar): #take x-Bar out
    x_axis = x_list.copy()
    for i in range(N+1):
        x_axis[i] = np.concatenate(x_axis[i])
        # np.append(x_axis[i], x_bar)
    for i in range (n):
        plt.plot(range(N+1), [x_axis[f][i] for f in range(N+1)], linewidth=3)
    dummy = np.full((N+1,1), xbar)
    # plt.plot(range(N+1), dummy, 'b--', linewidth=4)

def error_plt(errors, G, blks, sol, rate='arbi'):
    if rate == 'cliques':
        r = rate_cliques(G, blks)
        #blabel = r'Predicted Bound in Cor 1.3, $(1-\frac{r\alpha(G)}{4K})^k||c-c^*||^2$'
        label = 'Block Gossip (Cliques)'
    elif rate == 'ies':
        r = rate_ies(G,blks)
        #blabel =  r'Predicted Bound in Cor 1.3, $(1-\frac{r\alpha(G)}{2K})^k||c-c^*||^2$'
        label = 'Block Gossip (Independent Edge Sets)'
    elif rate == 'path':
        r = rate_paths(G, blks)
        #blabel = r'Predicted Bound in Cor 1.3, $(1-\frac{r\alpha(G)}{4K})^k||c-c^*||^2$'
        label = 'Block Gossip (Paths)'
    elif rate == 'arbi':
        r = rate_arbi(G, blks)
        #blabel = r'Predicted Bound in Cor 1.3, $(1-\frac{r\alpha(G)}{MK})^k||c-c^*||^2$'
        label = 'Block Gossip'
    else:
        print('rate not supported, using arbitrary blk rate')
        r = rate_arbi(G, blks)
    blabel = r'Predicted Bound'
    #bound = np.full((N+1), (r**i)*(errors[0]**2))
    bound = [(r**i)*(errors[0]**2) for i in range(N+1)]
    err = [errors[i]**2 for i in range(len(errors))]
    plt.semilogy(range(np.shape(errors)[0]),err, 'b', linewidth=4, label = r'Block RK')
    plt.semilogy(range(np.shape(bound)[0]), bound, 'r--', linewidth=4, label = blabel)
    plt.legend(prop={'size': 15})
    return bound
