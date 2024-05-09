import numpy as np
import gudhi
import matplotlib.pyplot as plt
import networkx as nx

"""
Class for Creating and Evaluating Feautures of Simplicial Complex
"""


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
        self.pairwise_distances = -1
        self.graph = -1
        self.simplices = [] # for k>=2
        
        for t in trajectories:
            self.add_trajectory(t)

    def reset(self):
        self.pairwise_distances = -1
        self.graph = -1
        self.simplices = -1 # for k>=2

    def construct_simplex(self, epsilon, store = True):

        if(type(self.pairwise_distances) == int):
            self.compute_pairwise_distances()

        new_edges = self.__connect_edges(epsilon)
        graph, new_simplex = self.__form_k_simplices(new_edges, epsilon, store)

        return graph, new_simplex

    def add_trajectory(self, trajectory, connect_frames=True):

        N, T = trajectory.shape
        assert N == self.N, "trajectory dimensionality does not match" 
        
        for i, t in enumerate(trajectory['T']):
            v = trajectory.sel(T=t).data
            v_index = len(self.vertices)
            self.vertices.append(v)
            if(i > 0 and connect_frames): # connect to previous state vector in trajectory
                self.base_edges.add((v_index-1, v_index))

        self.pairwise_distances = -1 # reset
        self.graph = -1
    
    def compute_pairwise_distances(self):
        vertices = np.array(self.vertices)
        V, D = vertices.shape
        pairwise_diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
        pairwise_distances = np.linalg.norm(pairwise_diff, axis=2)
        self.pairwise_distances = pairwise_distances
        
    def __connect_edges(self, epsilon, add=True):
        new_edges = []
        
        row_indices, col_indices = np.where(self.pairwise_distances < epsilon)
        for i, j in zip(row_indices, col_indices):
            if(i!=j):
                new_edges.append((i,j))
        if add:
            new_edges = self.base_edges.union(set(new_edges))
        return new_edges

    def __form_k_simplices(self, edges, epsilon, store = True):

        assert len(self.vertices) > 0, "Construct graph first"
        assert len(self.base_edges) > 0, "Construct graph first"

        graph = nx.Graph()
        for i in range(len(self.vertices)):
            graph.add_node(i, vector=self.vertices[i])
        graph.add_edges_from(edges)
        cliques = nx.find_cliques(graph)
        extract_simplices = {}

        for c in cliques:
            k = len(c)
            if(k>2): 
                if(k not in extract_simplices.keys()):
                    extract_simplices[k] = []
                extract_simplices[k].append(c)

        if store:
            self.simplices.append(extract_simplices)
            self.graph.append(graph)
        return (graph, extract_simplices)

    def draw(self):
        nx.draw(self.graph, with_labels=True, node_color='skyblue', node_size=200, font_size=12)
        plt.show()

    def persist_precomputed(self, k, plot = True):

        if(type(self.pairwise_distances) == int):
            pairwise_distances = self.compute_pairwise_distances() 

        rc = gudhi.RipsComplex(distance_matrix = pairwise_distances)
        st = rc.create_simplex_tree(max_dimension=k)   

        diagram = st.persistence()
        if plot:
            gudhi.plot_persistence_diagram(diagram)
            plt.show()
        return diagram, st
    
    def persist_complex(self, max_epsilon, plot = True):

        st = gudhi.SimplexTree()
        E = np.arange(start = 0, stop = max_epsilon, step = max_epsilon/20)

        for e in E:
            graph, s = self.construct_simplex(epsilon = e, store = False)
            filtration_val = np.ones((len(s), ))*e 
            st.insert_batch(s, filtration_val)

        diagram = st.persistence()
        if plot:
            gudhi.plot_persistence_barcode(diagram)
            plt.show()

            gudhi.plot_persistence_diagram(diagram)
            plt.show()
        return diagram, st
    
    def contains(self, p, st : gudhi.SimplexTree, st_type : str):
        
        assert st_type == 'complex' or st_type == 'precomputed', "Invalid SimplexTree type"

        if st_type == 'complex':
            

        if st_type == 'precomputed':

            
    def boundary_matrix(self):
        pass