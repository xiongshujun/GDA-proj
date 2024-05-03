from gtda.homology import VietorisRipsPersistence

################################
#      PERSISTENT HOMOLOGY     #
################################

def persist(X, epsilon, hom, plot = False):
    """
    INPUTS
        X := Vietoris-Rips complex with inherent scale epsilon
        epsilon := feature scale epsilon on X, used here to improve ease of analysis
        hom := homology scales (0, 1, ..., k), where each one corresponds to a dimension of "hole" to look at
        plot := if True, then apply fit_transform_plot

    OUTPUTS
        fit_x: ndarray of shape (n_samples, n_features, 3) that plotting can fit over

    REFERENCES
        Homology Calculation: https://giotto-ai.github.io/gtda-docs/latest/modules/homology.html

    !TODO
        NEED TO MODIFY THIS FUNCTION TO ALLOW FOR TRAJECTORYMAP GUARANTEED EDGES
    """

    persistence = VietorisRipsPersistence(metric="euclidean", homology_dimensions=hom, n_jobs=5)
    
    if plot:
        fit_X = VietorisRipsPersistence.fit_transform_plot(X)
    else: 
        fit_X = persistence.fit_transform(X)
            # returns: an array of persistence diagrams computed from X
            # return type: ndarray of shape (n_samples, n_features, 3)
    
    return fit_X, epsilon

##########################
#   SPECTRAL CLUSTERING  #
##########################








###########################
#    EPSILON TIGHTENING   #
###########################


