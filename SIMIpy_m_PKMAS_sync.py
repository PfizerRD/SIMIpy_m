# SIMI Motion Gait Metrics (SIMIpy_m)
# DMTI PfIRe Lab
# Author: Visar Berki, Hao Zhang

# Determine timing of PKMAS sync signal from GS PKMAS.csv files
# Input: filepath_GS_sync
# Example: C:/.../X9001262 - Data/10010004\X9001262_A_10010004_02_Carpet_PKMAS_Sync.csv


# Library definitions:
import pandas
import numpy as np

# %% Load GS PMKAS as DataFrame format

def _get_PKMAS_sync (filepath):
    GS_PKMAS_data = pandas.read_csv(filepath, delimiter=',')
    skip = GS_PKMAS_data.iloc[:, 4] == 'Sync. In'
    skip = np.array([i for i, x in enumerate(skip) if x])
    skip=int(skip)
    GS_PKMAS_data = pandas.read_csv(filepath, delimiter=',', skiprows=skip+1)
    
    del skip
    
    PKMAS_sync_idx = np.where(GS_PKMAS_data['Sync. In'] == 1)
    PKMAS_sync_idx = PKMAS_sync_idx[0][0]
    
    PKMAS_sync_t = GS_PKMAS_data['Time (sec.)'][PKMAS_sync_idx]
    
    return GS_PKMAS_data, PKMAS_sync_t
