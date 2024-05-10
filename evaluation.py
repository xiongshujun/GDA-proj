import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import gudhi
from sklearn.cluster import SpectralClustering

from SimplicialComplex import SimplicialComplex


################################
#      PERSISTENT HOMOLOGY     #
################################

def persist_precomputed(W, k, plot = True):

    """
    Takes in distances at feature scale epsilon and returns a persistence diagram

    INPUTS
        W := precomputed distance matrix
        k := maximum homology dimension we're interested in
        plot := whether or not to plot the barcode

    OUTPUTS
        diagram := an array of persistence diagrams computed from X
                    type : list of pairs(dimension, pair(birth, death))
        st := SimplexTree made from the VR complex defined by W (precomputed distance matrix)

    """

    rc = gudhi.RipsComplex(distance_matrix = W)
    st = rc.create_simplex_tree(max_dimensions = k)

    diagram = st.persistence()
    if plot:
        gudhi.plot_persistence_barcode(diagram)
        plt.show()

        gudhi.plot_persistence_diagram(diagram)
        plt.show()
    return diagram, st

def persist_complex(S, E, plot = True):
    """
    INPUTS
        S := list of simplicial complexes where S[i] is formed from feature scale E[i]
            S[i] is defined by a set of new simplices of homogeneous dimension n formed at feature scale E[i]
                type(S[i]) := ndarray of shape (k, n), where we are inserting k n-simplices
                                S[i][i_prime] consists of a list of n integers denoting the vertices of the simplex

            S[0] is determined by the simplices generated by the inherent TrajectoryMap edges
        E := list of increasing feature scales s.t. i < j ==> E[i] <= E[j]
            if E[i] == E[j], that means S[i] and S[j] have the same feature scale but consist of simplices of different dimension
            !NOTE: E[0] = 0

        !NOTE: len(S) needs to equal len(E)
    
    OUTPUTS
        diagram := an array of persistence diagrams computed from X
                    type : list of pairs(dimension, pair(birth, death))
        st := SimplexTree formed by S, E pairs
    REFERENCES
        https://gudhi.inria.fr/python/latest/simplex_tree_ref.html
    """

    st = gudhi.SimplexTree()

    for i in range(len(S)):

        n = len(S[i][0])

        filtration_val = np.ones((n, ))*E[i]
        st.insert_batch(S[i], filtration_val)

    diagram = st.persistence()
    if plot:
        gudhi.plot_persistence_barcode(diagram)
        plt.show()

        gudhi.plot_persistence_diagram(diagram)
        plt.show()
    return diagram, st


##########################
#   SPECTRAL CLUSTERING  #
##########################

def cluster(X, assign_labels = 'discretize'):
    """
    Wrapper for scikit.learn.SpectralClustering

    INPUTS
        X := ndarray of original dataset
        k := number of classes
        assign_labels := passed 'assign_labels' parameter
                            for more info, refer to https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html

    OUTPUTS
        labels := labels for each of the datapoints in order
        clustering := post-fit SpectralClustering object
    """

    clustering = SpectralClustering(n_clusters=2,
                                        assign_labels=assign_labels).fit(X)
    
    return clustering.labels_, clustering

###########################
#    EPSILON TIGHTENING   #
###########################

"""
The central idea behind epsilon tightening is that TrajectoryMap provides the *better* homology information than standard VR complexes.
But what does "better" actually mean? Here, we propose that 
    1) The same core homology information can be extracted from a TrajectoryMap complex than a VR complex at a much lower epsilon bound.
            This also implies that we can denoise much more effectively with a TrajectoryMap
    2) TrajectoryMap complexes are more *robust* at the same epsilon. In other words, given a TrajectoryMap complex and a VR complex both formed up to filtration value epsilon,
            when new points not yet seen are evaluated, they are much more likely to lie within or closer to the TrajectoryMap complex compared to the VR complex

To test the former, we just need to look at (and hopefully quantify) differences between persistent homology diagrams of TrajectoryMaps and VR complexes.

To test the latter, we form partial complexes on training sets then evaluate the accuracy of predictions on test sets. 
"""

def epsilon_tighten(sc, Y, max_epsilon = 1, plot = False):
    """
    Given two simplicial complexes at a given epsilon value, evaluate how well-contained each point in a test set is within the complex by looking at
        1) How much the homology changes
            A) via differences in Betti numbers
            B) via differences in persistent homology diagrams (e.g. barcodes) --> this only happens when plot == True
        2) Does this point fit inside a face of a lower-level simplex?
            !NOTE: in future work, this can also be quantified better by measuring distance from the nearest face
    
    INPUTS
        sc := train dataset as an xr.DataArray, list of vectors that represent activity
        Y := test dataset as an xr.DataArray, list of vectors that represent activity
        epsilon := feature scale chosen to build the complexes

    INTERMEDIARY VARIABLES
        vr_complex := standard Vietoris-Rips complex in the form of a simplicial tree
        tm_complex := TrajectoryMap complex in the form of a simplicial tree
    
    OUTPUTS
        betti_diff := <Betti numbers of the VR complex> - <Betti numbers of the TM complex> (w/ accounted padding)
        acc_vr     := percentage accuracy of the VR complex in accounting for test set points
        acc_tm     := percentage accuracy of the TrajectoryMap complex in accounting for test set points
    """

    acc_vr = 0
    acc_tm = 0

    # GET COMPLEXES FROM sc    
    vr_diag, vr_complex = sc.persist_precomputed(max_epsilon, plot)
    tm_diag, tm_complex = sc.persist_complex(max_epsilon, plot)

    vr_complex.compute_persistence()
    tm_complex.compute_persistence()

    betti_vr = vr_complex.betti_numbers()
    betti_tm = tm_complex.betti_numbers()

    while len(betti_vr) < len(betti_tm):
        betti_vr.append(0)
    while len(betti_tm) < len(betti_vr):
        betti_tm.append(0)

    betti_diff = betti_vr - betti_tm

    for i in range(len(Y)):
        
        if sc.contains(Y[i], vr_complex, 'precomputed'):
            acc_vr += 1
        if sc.contains(Y[i], tm_complex, 'st_type'):
            acc_tm += 1
    
    acc_vr /= len(Y)
    acc_tm /= len(Y)
        
    return betti_diff, acc_vr, acc_tm