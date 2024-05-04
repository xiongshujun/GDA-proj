"""
General Class Container for Neural Data
"""
import os
import scipy.io as sio
import numpy as np


class zfSessionData:
    def __init__(self, filename):
        self.base_dir = "~/Documents/Geometric DA/Final Project/RamirezPaperData/analyzedBehaviorAndImaging"
        
        self.traces = os.path.join(self.base_dir, "CaTracesAndFootprints/")
        self.behavior = os.path.join(self.base_dir, "behavior10to20Hz/")

        data = sio.loadmat(os.path.join(self.traces, filename))
        self.cellLocs = [file[0] for file in data['localCoordinates']]
        self.fluor = [file[0] for file in data['fluorescence']]
        self.num_samples = self.fluor[0].shape[1] # num_cells x num_samples
    
    def load_trajectory(self, idx):
        # idx belongs to different OG tiff files (RHB vs non RHB?), likely z planes
        fluor = self.fluor[idx]
        # dF/F
        temporal_mean = np.mean(fluor, axis=1).reshape(-1, 1)
        dFF = (fluor - temporal_mean) / temporal_mean
        return(dFF)


