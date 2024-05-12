import scipy.io as sio
import numpy as np
import os
import pandas as pd

import xarray as xr
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns


class zfSessionDataMotor:
    def __init__(self, fish=[], method='ep'):
        self.base_dir = "/Users/samart/Documents/Geometric DA/Final Project/RamirezPaperData/analyzedBehaviorAndImaging"        
        self.traces = os.path.join(self.base_dir, "CaTracesAndFootprints/")
        self.behavior = os.path.join(self.base_dir, "behavior10to20Hz/")
        # A is output of CaImAn, Gaussian Centered around cell locations
        # A binary is a binary mask that tells where cells are; A itself pretty much useless

    def get_filename(self, fish):
        if(self.method == 'ep'):
            return(f"{fish}_traces_EPSelect.mat")
        elif(self.method == 'im'):
            return(f"{fish}_traces_IMOpenSelect.mat")
        
    def parse_trace(self, fish):
        filename = self.get_filename(fish)
        data = sio.loadmat(os.path.join(self.traces, filename))
        cellLocs = [file[0] for file in data['localCoordinates']]
        fluor = [file[0] for file in data['fluorescence']]
        num_samples = fluor[0].shape[1] # num_cells x num_samples
        file_data = {
            'fluor' : fluor,
            'cellLocations' : cellLocs,
            'num_samples' : num_samples,
        }
        self.data[fish] = file_data
        return(data)

    def load_behavior(self, fish, idx):
        fname = os.path.join(self.behavior, f"{fish}_{idx}.mat")
        data = sio.loadmat(fname)
        return(data)

    def load_motor_trajectory(self, norm=False):
        df = pd.read_csv("motor_data.csv").drop(['Unnamed: 0'], axis=1)
        t, N = df.values.shape # num_samples x num_cells
        da = xr.DataArray(df.values, dims=('T', 'N'), coords={'N' : np.arange(1, N+1), 'T' : np.arange(0, t)})
        if(norm):
            da = (da - da.mean(dim='T')) / da.mean(dim='T')
        return(da)
    
    def load_fish_trajectory(self, fish, idx):
        # idx belongs to different OG tiff files (RHB vs non RHB?), likely z planes
        fluor = self.data[fish]['fluor'][idx]
        loc = self.data[fish]['cellLocations'][idx]
        # dF/F
        temporal_mean = np.mean(fluor, axis=1).reshape(-1, 1)
        dFF = (fluor - temporal_mean) / temporal_mean

        N, t = dFF.shape
        da = xr.DataArray(dFF, dims=('N', 'T'), coords={'N' : np.arange(1, N+1), 'T' : np.arange(0, t), 
                                                        'X' : (('N',), loc[:,0]), 'Y' : (('N',), loc[:,1])})
        
        return(da)