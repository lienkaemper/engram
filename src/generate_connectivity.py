# -*- coding: utf-8 -*-

import numpy as np


def macro_weights(J, h3, h1, g, h_i =1, g_ii = 1, h_i_ca3 = 1):
    '''
    regional weights for model of hippocampus

    J -- overall connectivity strength
    h3 -- engram-engram weight (CA3)
    h1 -- engram-engram weight (CA1)
    g -- inhibitory - excitatory weight
    h_i -- relative strength of inhibition onto engram cells (CA1)
    g_ii -- inhibition onto inhibition 
    h_i_ca3 -- relative strength of inhibition onto engram cells (CA3)
    '''
    return J*np.array([[ h3, 1, -h_i_ca3*g, 0, 0, 0], #CA3E
                        [1,  1, -g, 0, 0, 0], #CA3P
                        [1,  1, -g_ii, 0, 0, 0],  #CA3I
                        [h1, 1,  0, 0, 0, -h_i*g], #CA1E 
                        [1,  1,  0, 0, 0, -g],  #CAIP
                        [2,  2,  0, 1, 1, -g_ii]]) #CA1I

def weights_from_regions(index_dict, adjacency, macro_weights):
    '''
    produce neuron-level weight matrix from index dictionary, adjacency matrix, and regional weights

    index_dict -- dictionary where keys are region names and values are iterables indexing the neurons in each region
    adjacency -- neuron by neuron adjacency matrix, already scaled by 1/N
    macro_weights -- regional connectivity
    '''
    JJ = []
    for i, region_i in enumerate(index_dict):
        row = []
        for j, region_j in enumerate(index_dict):
            A_loc = adjacency[np.ix_(index_dict[region_i], index_dict[region_j])]
            J_ij = A_loc * macro_weights[i,j]
            row.append(J_ij)
        row =np.concatenate(row, axis = 1)
        JJ.append(row)
    JJ = np.concatenate(JJ, axis = 0)
    return JJ


def gen_adjacency(cells_per_region, macro_connectivity, regions = ["CA3E", "CA3P", "CA3I", "CA1E", "CA1P", "CA1I"]):
    '''
    generate neuron-level adjacency matrix and region-to-neuron index dictionary
    
    cells_per_region -- number of cells for each region
    macro_connectivity -- connection probability for each pair of regions
    regions -- list of region names 
    '''
    N = np.sum(cells_per_region)
    index_dict = {}
    count = 0
    for i, region in enumerate(regions):
        index_dict[region] = range(count, count + cells_per_region[i])
        count +=  cells_per_region[i]
    JJ = []
    for i, n_i in enumerate(cells_per_region):
        row = []
        for j, n_j in enumerate(cells_per_region):
            J_ij = (np.random.rand(n_i, n_j) < macro_connectivity[i,j] )/(n_j * macro_connectivity[i,j] )
            row.append(J_ij)
        row = np.concatenate(row, axis = 1)
        JJ.append(row)
    JJ = np.concatenate(JJ, axis = 0)
    return JJ, index_dict


def hippo_weights(index_dict, adjacency, h3, h1, g, J, i_plast=1, i_plast_3 = 1, g_ii =1):
    ''' neuron-level weights for model of hippocampus
    
    index_dict -- dictionary where keys are region names and values are iterables indexing the neurons in each region
    adjacency -- neuron by neuron adjacency matrix, already scaled by 1/N
    h3 -- engram-engram weight (CA3)
    h1 -- relative strength of inhibition onto engram cells (CA1)
    g -- inhibitory - excitatory weight
    J -- overall connectivity strength
    i_plast -- relative strength of inhibition onto engram cells (CA1)
    i_plast_3 --  relative strength of inhibition onto engram cells (CA3)
    g_ii -- inhibition onto inhibition 
    '''

    A =  weights_from_regions(index_dict, adjacency, macro_weights( J = J, h3 = h3, h1 = h1, g = g,h_i = i_plast, h_i_ca3=i_plast_3, g_ii = g_ii) )
    return A
