import numpy  as np
import pandas as pd
import xarray as xr

############################
#      DATA PROCESSING     #
############################
df = pd.read_csv("motor_data.csv").drop(['Unnamed: 0'], axis=1)
t, N = df.values.shape # num_samples x num_cells
da = xr.DataArray(df.values.T, dims=('N', 'T'), coords={'N' : np.arange(1, N+1), 'T' : np.arange(0, t)})



###########################
#        EVALUATION       #
###########################


# Build SimplicialComplex object



# Draw Persistent Homology diagrams



# Spectral Clustering comparisons




# Epsilon Tightening comparisons