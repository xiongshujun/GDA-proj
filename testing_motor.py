import numpy  as np
import gudhi
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

from Trajectory_motor import zfSessionDataMotor
from SimplicialComplex import SimplicialComplex
from evaluation import epsilon_tighten

############################
#      DATA PROCESSING     #
############################
zf = zfSessionDataMotor()
traj = zf.load_motor_trajectory(norm=True)

interval = 10
traj_sampled = traj.isel(T=slice(None, None, interval))

test_idx = np.arange(interval//2, len(traj[0]), interval)
traj_sampled_test = traj.isel(T=test_idx)

sc = SimplicialComplex([traj_sampled]) # Simplicial complex based on sampled data


###########################
#        EVALUATION       #
###########################

max_epsilon = 4

# PCA comparisons
    # Process is to PCA the data, then plot that PCA's homology diagram

# 2D PCA
pca_2 = PCA(n_components=2)
pca_2.fit(traj_sampled.data.T)
data_2d = pca_2.transform(traj_sampled.data.T)
rc = gudhi.RipsComplex(points=data_2d, max_edge_length = max_epsilon)
st = rc.create_simplex_tree(max_dimension=2)

diagram = st.persistence()
gudhi.plot_persistence_barcode(diagram)
plt.title("Persistence Barcode of 2D PCA")
plt.show()

gudhi.plot_persistence_diagram(diagram)
plt.title("Persistence Diagram of 2D PCA")
plt.show()


# 3D PCA
pca_3 = PCA(n_components=3)
pca_3.fit(traj_sampled.data.T)
data_3d = pca_3.transform(traj_sampled.data.T)
rc = gudhi.RipsComplex(points=data_3d, max_edge_length = max_epsilon)
st = rc.create_simplex_tree(max_dimension=3)

diagram = st.persistence()
gudhi.plot_persistence_barcode(diagram)
plt.title("Persistence Barcode of 3D PCA")
plt.show()

gudhi.plot_persistence_diagram(diagram)
plt.title("Persistence Diagram of 3D PCA")
plt.show()

# Epsilon Tightening
    # Persistent Homology diagrams drawn via plot = True

betti_diff, acc_vr, acc_tm = epsilon_tighten(sc, [traj_sampled_test], max_epsilon = max_epsilon, plot = True)

print("The difference in Betti values at different levels is: " + betti_diff)
print("The accuracy of Vietoris-Rips complexes is: " + acc_vr)
print("The accuracy of TrajectoryMap is: " + acc_tm)