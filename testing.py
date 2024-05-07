import numpy as np
import pandas as pd

################################
#        DATA PROCESSING       #
################################
df = pd.read_csv('motor_data.csv', sep=',', header=None)
motor = np.array(df.values)




# BUILDING A PCA MODEL 



# BUILDING AN IsoMap MODEL






# BUILDING A NAIVE SIMPLICIAL COMPLEX
    # Build a simplicial complex based off of vertices




# BUILDING A TRAJECTORYMAP COMPLEX
    # Build a simplicial complex based off of vertices *and* add additional edges as needed




# EVALUATION IN ACCORDANCE TO README