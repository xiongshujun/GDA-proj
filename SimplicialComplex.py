"""
Class for Creating and Evaluating Feautures of Simplicial Complex
"""
import numpy as np 
import networkx as nx
import xarray as xr
import gudhi
import matplotlib.pyplot as plt

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
        self.edges = {}
        self.graphs = {}
        self.simplices = {} # for k>=2
        
        for t in trajectories:
            self.add_trajectory(t)

        self.compute_pairwise_distances()

    def compute_pairwise_distances(self):
        vertices = np.array(self.vertices)
        V, D = vertices.shape
        pairwise_diff = vertices[:, np.newaxis, :] - vertices[np.newaxis, :, :]
        pairwise_distances = np.linalg.norm(pairwise_diff, axis=2)
        self.pairwise_distances = pairwise_distances

    def add_trajectory(self, trajectory, connect_frames=True, recompute_dists=False):
        N, T = trajectory.shape
        assert N == self.N, "trajectory dimensionality does not match" 
        self.edges['trajectory'] = set()
        for i in range(T):
            v = trajectory.isel(T=i).data
            v_index = len(self.vertices)
            self.vertices.append(v)
            if(i > 0 and connect_frames): # connect to previous state vector in trajectory
                self.edges['trajectory'].add((v_index-1, v_index))
        if(recompute_dists):
            self.compute_pairwise_distances()
        
    def connect_edges(self, epsilon):
        new_edges = []
        dists = np.triu(self.pairwise_distances, k=0) # ignore lower half
        row_indices, col_indices = np.where(np.logical_and(dists<epsilon, dists>0))
        for i, j in zip(row_indices, col_indices):
            if(i!=j):
                new_edges.append((i,j))
        self.edges[epsilon] = set(new_edges)

    def all_edges(self, epsilon):
        if(epsilon not in self.edges.keys()):
            print("Eps not found, connecting edges")
            self.connect_edges(epsilon)
        edges = self.edges[epsilon]  
        edges = edges.union(self.edges['trajectory'])
        return(edges)

    def form_graph(self, eps):
        graph = nx.Graph()
        for i,v in enumerate(self.vertices):
            graph.add_node(i, vector=v)
        edges = self.all_edges(eps)
        print(f"Forming graph with {len(self.vertices)} vertices and {len(edges)} edges")
        graph.add_edges_from(edges)
        self.graphs[eps] = graph
        return(graph)
        
    def form_k_simplices(self, epsilon, kmax):
        assert len(self.vertices) > 0, "Construct graph first"
        assert len(self.edges) > 0, "Construct graph first"
        if(epsilon not in self.graphs.keys()):
            graph = self.form_graph(epsilon)
        else:
            graph = self.graphs[epsilon]
        cliques = nx.find_cliques(graph)
        extract_simplices = {}
        for c in cliques:
            k = len(c)
            if(k>2 and k<= kmax): 
                if(k not in extract_simplices.keys()):
                    extract_simplices[k] = []
                extract_simplices[k].append(c)
        self.simplices[epsilon] = extract_simplices
        return(extract_simplices)

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

    def persist_precomputed(self, k, plot = True):
        rc = gudhi.RipsComplex(distance_matrix = self.pairwise_distances)
        st = rc.create_simplex_tree(max_dimension=k)   
        diagram = st.persistence()
        if plot:
            gudhi.plot_persistence_barcode(diagram)
            plt.title("Persistence Barcode")
            plt.show()
            gudhi.plot_persistence_diagram(diagram)
            plt.title("Persistence Diagram")
            plt.show()
        return diagram, st
    
    def persist_complex(self, plot = True):
        st = gudhi.SimplexTree()
        max_val = np.max(self.pairwise_distances)
        E = np.arange(start = 0, stop = max_val + 1, step = max_val/20)

        for e in E:
            graph, s = self.construct_simplex(epsilon = e, store = False)
            filtration_val = np.ones((len(s), ))*e 
            st.insert_batch(s, filtration_val)

        diagram = st.persistence()
        if plot:
            gudhi.plot_persistence_barcode(diagram)
            plt.title("Persistence Barcode")
            plt.show()
            gudhi.plot_persistence_diagram(diagram)
            plt.title("Persistence Diagram")
            plt.show()
        return diagram, st
            
    def boundary_matrix(self):
        pass
