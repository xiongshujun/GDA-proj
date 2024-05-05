import numpy as np
import matplotlib.pyplot as plt

import gudhi
from sklearn.cluster import SpectralClustering

################################
#      PERSISTENT HOMOLOGY     #
################################

def betti(X, epsilon, tolerance = 1e-5):
    """
    Takes in a canonically ordered simplicial complex of feature scale epsilon, returns the Betti numbers 

    INPUTS
        X := simplicial complex in canonical ordering
        epsilon := feature scale epsilon on X, used here to improve ease of analysis

    OUTPUTS
        betti := list of Betti numbers

    REFERENCES
        https://orangewire.xyz/mathematics/2022/03/25/simplicial-homology.html
    """

    def __boundary(X):
        """
        Defines the sequential boundary maps on X of increasing dimension

        Returns a list of boundary maps to be used in computing homology for Betti numbers
        """
        
        maps = []
        for s, s_prime in zip(X, X[1:]):

            sign = []

            for s_prime_minus in s_prime:

                faces   = []                
                for i in range(len(s_prime_minus)): # get the faces of a simplex by returning all possible substrings made by deleting one of the cahracters
                    faces.append(s_prime_minus[:i] + s_prime_minus[i+1:]) #!TODO: this is assuming each s_minus[:i] is a string!

                for s_prime in s:  # return (-1) to the power of parity if there is an overlap, 0 otherwise

                    if s_prime in faces:
                        idx = faces.index(s_prime)
                        sign.append((-1)**(idx//2))
                    else: sign.append(0)

            maps.append(np.array(sign).T)

        return maps
    
    def __homology(bd_maps, tolerance): # using the cokernel trick!

        # PADDING
        minus_pad = bd_maps[-1].shape[1]
        plus_pad  = bd_maps[0].shape[0]

        bd_maps.insert(0, np.ones(shape=(0, plus_pad)))
        bd_maps.append(np.ones(shape=(minus_pad, 0)))

        # KERNEL AND COKERNAL CALCULATION USING SVD, USING REFERNECE
        #!TODO: FIGURE OUT WHAT THIS ACTUALLY DOES!!!!!
        def __kernel(A):
            _, s, vh = np.linalg.svd(A)
            sing = np.zeros(vh.shape[0])
            sing[:s.size] = s
            null = np.compress(sing <= tolerance, vh, axis=0)
            return null.T
        
        def __cok(A):
            u, s, _ = np.linalg.svd(A)
            sing = np.zeros(u.shape[1])
            sing[:s.size] = s
            return np.compress(sing <= tolerance, u, axis=1)
        
        H = []
        for dbd_k, dbd_k_prime in zip(bd_maps, bd_maps[1:]):
            kappa = __kernel(dbd_k, tolerance)
            psi, _, _, _ = np.linalg.lstsq(kappa, dbd_k_prime, rcond=None)

            ksi = __cok(psi, tolerance)
            H.append(np.dot(kappa, ksi))

    bd_maps = __boundary(X)
    H = __homology(bd_maps, tolerance)
    betti = [basis.shape[1] for basis in H]
    return betti, epsilon

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

    REFERENCES

    """

    rc = gudhi.RipsComplex(distance_matrix = W)
    simplex_tree = rc.create_simplex_tree(max_dimensions = k)

    diagram = simplex_tree.persistence()
    if plot:
        gudhi.plot_persistence_barcode(diagram)
        plt.show()

        gudhi.plot_persistence_diagram(diagram)
        plt.show()
    return diagram

def persist_complex(S, E, plot = True):
    """
    INPUTS
        S := list of simplicial complexes where S[i] is formed from feature scale E[i]
            S[i] is defined by a set of new simplices of homogeneous dimension n formed at feature scale E[i]
                type(S[i]) := numpy.array of shape (k, n), where we are inserting k n-simplices
                                S[i][i_prime] consists of a list of n integers denoting the vertices of the simplex

            S[0] is determined by the simplices generated by the inherent TrajectoryMap edges
        E := list of increasing feature scales s.t. i < j ==> E[i] <= E[j]
            if E[i] == E[j], that means S[i] and S[j] have the same feature scale but consist of simplices of different dimension
            !NOTE: E[0] = 0

        !NOTE: len(S) needs to equal len(E)
    
    OUTPUTS
        diagram := an array of persistence diagrams computed from X
                    type : list of pairs(dimension, pair(birth, death))
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
    return diagram


##########################
#   SPECTRAL CLUSTERING  #
##########################

def cluster(X, k, assign_labels = 'discretize'):
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
