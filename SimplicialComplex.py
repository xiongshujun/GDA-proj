
"""
Class for Creating and Evaluating Feautures of Simplicial Complex
"""
import numpy as np 
import networkx as nx
import xarray as xr
import gudhi
import matplotlib.pyplot as plt
from collections import defaultdict

from scipy.spatial import KDTree
from itertools import chain, combinations

# courtesy of https://stackoverflow.com/questions/1482308/how-to-get-all-subsets-of-a-set-powerset
def powerset(s):
    x = len(s)
    masks = [1 << i for i in range(x)]
    for i in range(1 << x):
        yield [ss for mask, ss in zip(masks, s) if i & mask] 

class SimplicialComplex:
    def __init__(self, trajectories):
        """
        Input: 
            trajectories is a list of neural time series of N dimensional vectors (Nxti) matrix). Each time series is xarray data array
            Time Series assumed to lie on k<N dimensional manifold
            Time series length does not have to be the same, N has to be same for trajectories to exist in same space
            I matrices of size N by Ti 
        """
        self.N = trajectories[0].shape[0]
        self.trajectories = trajectories
        self.vertices = []
        self.base_edges = set()
        self.homology = defaultdict(dict)

        for t in trajectories:
            self.add_trajectory(t)

        self.compute_pairwise_distances()


    def compute_pairwise_distances(self):
        vertices = np.array(self.vertices)
        V, D = vertices.shape
        pairwise_diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
        pairwise_distances = np.linalg.norm(pairwise_diff, axis=2)
        self.pairwise_distances = pairwise_distances

    def add_trajectory(self, trajectory, connect_frames=True, recompute_dists=True):
        N, T = trajectory.shape
        assert N == self.N, "trajectory dimensionality does not match" 
        for i in range(T):
            v = trajectory.isel(T=i).data
            v_index = len(self.vertices)
            self.vertices.append(v)
            if(i > 0 and connect_frames): # connect to previous state vector in trajectory
                self.base_edges.add((v_index-1, v_index))
        if(recompute_dists):
            self.compute_pairwise_distances()
        
    def connect_edges(self, epsilon):
        new_edges = []
        dists = np.triu(self.pairwise_distances, k=0) # ignore lower half
        row_indices, col_indices = np.where(np.logical_and(dists<epsilon, dists>0))
        for i, j in zip(row_indices, col_indices):
            if(i!=j):
                new_edges.append((i,j))
        self.homology[epsilon]['edges'] = set(new_edges)
        return(new_edges)

    def all_edges(self, epsilon):
        if('edges' not in self.homology[epsilon]):
            print("Eps not found, connecting edges")
            self.connect_edges(epsilon)
        edges = self.homology[epsilon]['edges']
        edges = edges.union(self.base_edges)
        return(edges)

    def form_graph(self, eps):
        graph = nx.Graph()
        for i,v in enumerate(self.vertices):
            graph.add_node(i, vector=v)
        edges = self.all_edges(eps)
        print(f"Forming graph with {len(self.vertices)} vertices and {len(edges)} edges")
        graph.add_edges_from(edges)
        self.homology[eps]['graph'] = graph
        return(graph)
        
    def form_simplices(self, epsilon, kmax):

        if('graph' in self.homology[epsilon]):
            graph = self.homology[epsilon]['graph']
        else:
            graph = self.form_graph(epsilon)
            
        cliques = nx.find_cliques(graph)
        extract_simplices = {}
        tree = gudhi.SimplexTree()
        for k in range(2, kmax+1):
            extract_simplices[k] = []
        for c in cliques:
            c.sort()
            k = len(c) - 1 # k simplex has k+1 simplices
            if(k>1 and k<=kmax):
                tree.insert(c, filtration=epsilon)
                extract_simplices[k].append(c)
        self.homology[epsilon]['simplex_sets'] = extract_simplices
        self.homology[epsilon]['tree'] = tree
        return(tree)

    def draw_graph(self, graph):
        nx.draw(graph, with_labels=True, node_color='skyblue', node_size=200, font_size=12)
        plt.show()

    def contains(self, p, nearest_k, st : gudhi.SimplexTree, st_type : str):
        assert st_type == 'complex' or st_type == 'precomputed', "Invalid SimplexTree type"
        """

        DEPRECATED METHOD
        p is contained in a given simplex if the vector from one vertex to it is a linear combination of 
            geometrically independent basis vectors form by the sides of the face such that all coefficients are in the interval [0, 1]
        we only look at simplices formed by the nearest k datapoints

        """
        df = self.trajectories[0].data
        distances, indices = KDTree(df).query(p, k = nearest_k)

        # for each set made up of the ordered indices made of 'indices', check if such a simplex exists in st using .find()
        p_set = list(powerset(indices))
        for s in p_set:
            if st.find(s): return True                

        """
        COOL METHOD BELOW THAT UNFORTUNATELY IS NP-HARD
        for s, f in simplices:
            if len(s) > 1:
                M = []
                p_0 = p - s[0]
                for i in range(1, len(s)):
                    M.append(df[s[i]] - df[s[0]])
                M = np.array(M).T
                predict = np.linalg.lstsq(M, p_0) # using least-squares regression to see if it fits
                coeffs = predict[0]
                res = predict[1]
                if np.max(res) < f/2 and np.min(res) > -1*f/2 and np.max(coeffs) <= 1 and np.min(coeffs) >= 0:
                    return True
        """

        return False

    def simplex_from_distances(self, max_epsilon, kmax):
        rc = gudhi.RipsComplex(distance_matrix = self.pairwise_distances, max_edge_length = max_epsilon)
        st = rc.create_simplex_tree(max_dimension=kmax)   
        return(st)

    def plot_persistence(self, name, simplex_tree, plot=True):

        diagram = simplex_tree.persistence()
        if(plot):
            gudhi.plot_persistence_barcode(diagram)
            plt.title("Persistence Barcode using " + name)
            plt.show()
            gudhi.plot_persistence_diagram(diagram)
            plt.title("Persistence Diagram using " + name)
            plt.show()
        return(diagram)
    
    def persist_precomputed(self, max_epsilon, k = 116, plot = True):
        st = self.simplex_from_distances(max_epsilon, k)
        diagram = self.plot_persistence("Vietoris-Rips", st, plot)
        return diagram, st
    
    def persist_VR(self, max_epsilon, k = 116, plot = True):

        rc = gudhi.RipsComplex(points = self.trajectories[0].data, max_edge_length = max_epsilon)
        st = rc.create_simplex_tree(max_dimension = len(self.trajectories[0].data.T))
        diagram = self.plot_persistence("Vietoris-Rips", st, plot)
        return diagram, st

    def persist_trajectoryMap(self, max_epsilon, kmax = 5, plot = True):
        st = self.form_simplices(max_epsilon, kmax)
        diagram = self.plot_persistence("TrajectoryMap", st, plot)
        return diagram, st
    
    def persist_complex(self, max_epsilon, k = 5, fidelity = 1000, plot = True):

        st = gudhi.SimplexTree()
        E = np.arange(start = 0, stop = max_epsilon, step = max_epsilon/fidelity)

        for e in E:

            sub_st = self.form_simplices(e, k)
            for s, f in sub_st.get_simplices():
                st.insert(s, f)

        diagram = self.plot_persistence("TrajectoryMap", st, plot)
        return diagram, st
            
    def boundary_matrix(self):
        pass