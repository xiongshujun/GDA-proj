import numpy as np
from gtda.homology import VietorisRipsPersistence

################################
#      PERSISTENT HOMOLOGY     #
################################

def betti(X, epsilon, tolerance = 1e-5):
    """
    INPUTS
        X := simplicial complex in canonical ordering
        epsilon := feature scale epsilon on X, used here to improve ease of analysis

    OUTPUTS
        betti := list of Betti numbers

    REFERENCES
        https://orangewire.xyz/mathematics/2022/03/25/simplicial-homology.html

    !TODO
        NEED TO MODIFY THIS FUNCTION TO ALLOW FOR TRAJECTORYMAP GUARANTEED EDGES
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
    H = __homology(bd_maps)
    return [basis.shape[1] for basis in H], epsilon

"""
    persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=hom, n_jobs=5)
    
    if plot:
        fit_X = VietorisRipsPersistence.fit_transform_plot(X)
    else: 
        fit_X = persistence.fit_transform(X)
            # returns: an array of persistence diagrams computed from X
            # return type: ndarray of shape (n_samples, n_features, 3)
    
    return fit_X, epsilon
"""

##########################
#   SPECTRAL CLUSTERING  #
##########################








###########################
#    EPSILON TIGHTENING   #
###########################


