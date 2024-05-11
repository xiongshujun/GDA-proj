
"""
Class for Creating and Evaluating Feautures of Simplicial Complex
"""
import numpy as np 
import networkx as nx
import xarray as xr
import gudhi
import matplotlib.pyplot as plt
from collections import defaultdict

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

    def contains(self, p, st : gudhi.SimplexTree, st_type : str):
        assert st_type == 'complex' or st_type == 'precomputed', "Invalid SimplexTree type"
        """

        p is contained in a given simplex if the vector from one vertex to it is a linear combination of 
            geometrically independent basis vectors form by the sides of the face such that all coefficients are in the interval [0, 1]

        """
        simplices = st.get_simplices()
            # returns generator with tuples(simplex, filtration)
        for s, f in simplices:
            if len(s) > 1:
                M = []
                p_0 = p - s[0]
                for i in range(1, len(s)):
                    M.append(self.trajectories(s[i]) - self.trajectories(s[0]))
                M = np.array(M).T
                coeffs, res = np.linalg.lstsq(M, p_0, rcond=None) # using least-squares regression to see if it fits
                if np.max(res) < f/2 and np.min(res) > -1*f/2 and np.max(coeffs) <= 1 and np.min(coeffs) >= 0:
                    return True
        return False

    def simplex_from_distances(self, kmax):
        rc = gudhi.RipsComplex(distance_matrix = self.pairwise_distances)
        st = rc.create_simplex_tree(max_dimension=kmax)   
        return(st)

    def plot_persistence(self, simplex_tree, plot=True):
        diagram = simplex_tree.persistence()
        if(plot):
            gudhi.plot_persistence_barcode(diagram)
            plt.title("Persistence Barcode using Vietoris-Rips")
            plt.show()
            gudhi.plot_persistence_diagram(diagram)
            plt.title("Persistence Diagram using Vietoris-Rips")
            plt.show()
        return(diagram)
    
    def persist_complex(self, max_epsilon, k = 5, plot = True):

        st = gudhi.SimplexTree()
        E = np.arange(start = 0, stop = max_epsilon, step = max_epsilon/20)

        for e in E:
            s = self.form_k_simplices(epsilon = e, kmax = k)
            filtration_val = np.ones((len(s), ))*e 
            st.insert_batch(s, filtration_val)

        diagram = st.persistence()
        if plot:
            gudhi.plot_persistence_barcode(diagram)
            plt.title("Persistence Barcode using TrajectoryMap")
            plt.show()
            gudhi.plot_persistence_diagram(diagram)
            plt.title("Persistence Diagram using TrajectoryMap")
            plt.show()
        return diagram, st
            
    def boundary_matrix(self):
        pass