"""
Class for Creating and Evaluating Feautures of Simplicial Complex
"""


class SimplicialComplex:

    def __init__(self, trajectory):
        # the trajectory as a N x T matrix
        self.trajectory = trajectory
        self.vertices = self.compute_vertices(trajectory)
        self.edges = self.compute_edges(trajectory)

    def boundary_matrix(self):

