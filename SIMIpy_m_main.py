# SIMI Motion Gait Metrics (SIMIpy-m)
# DMTI PfIRe Lab
# Authors: Visar Berki, Hao Zhang

# Definitions

from bisect import bisect_left
import openpyxl
import os
import pandas
import numpy as np
from SIMIpy_m_filenames import Filenames
from SIMIpy_m_processing_filepair import _filepair_chooser
import SIMIpy_m_filenames
import importlib
import matplotlib as plt
importlib.reload(SIMIpy_m_filenames)

path = os.getcwd()
os.chdir(path)

Processing_filepairs = _filepair_chooser(Filenames)

# %% Save all processed file pairs for one participant into "Batch_Outputs" dictionary

Batch_Outputs = {
    "Filenames": [],
    "GS_calc": [],
    "GS_HS": [],
    "GS_PKMAS_sync": [],
    "SIMI_metrics": [],
    "Processing_filepairs": [],
    "FVA_vals_HS": [],
    "FVA_vals_TO": []
}


def _output_participant_trial(key):
    Batch_Outputs[key] = {
        "Normal": [],
        "Fast": [],
        "Slow": [],
        "Carpet": []
    }


for key in list(Batch_Outputs):
    _output_participant_trial(key)

Batch_Outputs['Filenames'] = Filenames
Batch_Outputs['trials'] = []

for n in range(0, len(Processing_filepairs)):

    filepairs = Processing_filepairs[n]

    if 'Normal' in filepairs[0]:
        trial = 'Normal'
    elif 'Fast' in filepairs[0]:
        trial = 'Fast'
    elif 'Slow' in filepairs[0]:
        trial = 'Slow'
    elif 'Carpet' in filepairs[0]:
        trial = 'Carpet'

    Batch_Outputs['trials'].append(trial)

    for file in range(0, len(filepairs)):
        index = filepairs[file].find('Sync')

        if filepairs[file].endswith(".txt"):
            filepath = filepairs[file]
        elif index != -1:
            filepath_GS_sync = filepairs[file]
        else:
            filepath_GS = filepairs[file]

        # filepath = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_SIMIMCS1_Normal_Walk_marker_processed_7_13_2021.txt"
        # filepath_GS = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS.csv"
        # filepath_GS_sync = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS_sync.csv"

    exec(open("SIMIpy_m_metrics.py").read())

    Batch_Outputs['Processing_filepairs'][trial] = filepairs
    Batch_Outputs['SIMI_metrics'][trial] = SIMI_metrics
    Batch_Outputs['GS_calc'][trial] = GS_calc
    Batch_Outputs['GS_HS'][trial] = GS_calc['First_Contact_time']
    Batch_Outputs['GS_PKMAS_sync'][trial] = GS_PKMAS_sync
    Batch_Outputs['FVA_vals_HS'][trial] = FVA_vars.HS
    Batch_Outputs['FVA_vals_TO'][trial] = FVA_vars.TO

    # del(SIMI_metrics, GS_calc, GS_PKMAS_sync)

# %% Save participant trials metrics into one spreadsheet

# Metrics for one filepair (one set of GS & SIMI Metrics Files)
# Metrics to compile:
    # 1. GS_Calc['Stride_Velocity_mean']
    # 2. GS_Calc['Speed_Both_Feet_mean']
    # 3. SIMI_metrics['Spine_SIMI_v_masked_mean']
    # 4. SIMI_metrics['Spine_pos_deriv_velocity_mean']
    # 5. SIMI_metrics['Velocity_Passes_mean']
    # 6. SIMI_metrics['Spine_pos_deriv_velocity_median']


# Nested metrics format (per participant):
Batch_Outputs['SIMI_metrics'][Batch_Outputs['trials'][0]]


def _output_metrics_compiled(Batch_Outputs):
    # participant number
    # pnum = Batch_Outputs['Filenames']['participant_num']
    trials = Batch_Outputs['trials']
    numtrials = len(trials)

    out_dat = np.zeros((numtrials, 6))
    # out_events = []

    for n in range(0, numtrials):

        # creating a numpy array
        out_dat[n, 0] = Batch_Outputs['GS_calc'][trials[n]]['Stride_Velocity_mean']
        out_dat[n, 1] = Batch_Outputs['GS_calc'][trials[n]]['Stride_Velocity_Passes_mean']
        out_dat[n, 2] = Batch_Outputs['SIMI_metrics'][trials[n]]['Spine_SIMI_v_masked_mean']
        out_dat[n, 3] = Batch_Outputs['SIMI_metrics'][trials[n]]['Spine_pos_deriv_velocity_mean']
        out_dat[n, 4] = Batch_Outputs['SIMI_metrics'][trials[n]]['Velocity_Passes_mean']
        out_dat[n, 5] = Batch_Outputs['SIMI_metrics'][trials[n]]['Spine_pos_deriv_velocity_median']

        # out_events[n, :] = Batch_Outputs['FVA_vals_HS'][trials[n]]

    return out_dat


Batch_Outputs['output_metrics_spreadsheet'] = _output_metrics_compiled(
    Batch_Outputs)

Batch_Outputs['output_metrics_spreadsheet'] = pandas.DataFrame(data=Batch_Outputs['output_metrics_spreadsheet'],
                                                               index=[Batch_Outputs['trials'][0],
                                                                      Batch_Outputs['trials'][1],
                                                                      Batch_Outputs['trials'][2],
                                                                      Batch_Outputs['trials'][3]],
                                                               columns=["GS_Stride_Velocity_mean",
                                                                        "GS_Stride_Velocity_Passes_mean",
                                                                        "SIMI_Spine_v_masked_mean",
                                                                        "SIMI_Spine_pos_deriv_velocity_mean",
                                                                        "SIMI_Velocity_Passes_mean",
                                                                        "SIMI_Spine_pos_deriv_velocity_median"])

# %% EVENT METRICS SPREADSHEET (HS)

df1 = pandas.DataFrame(Batch_Outputs['FVA_vals_HS']['Normal'])
df2 = pandas.DataFrame(Batch_Outputs['FVA_vals_HS']['Fast'])
df3 = pandas.DataFrame(Batch_Outputs['FVA_vals_HS']['Slow'])
df4 = pandas.DataFrame(Batch_Outputs['FVA_vals_HS']['Carpet'])

df5 = pandas.DataFrame(Batch_Outputs['GS_HS']['Normal'])
df6 = pandas.DataFrame(Batch_Outputs['GS_HS']['Fast'])
df7 = pandas.DataFrame(Batch_Outputs['GS_HS']['Slow'])
df8 = pandas.DataFrame(Batch_Outputs['GS_HS']['Carpet'])
df = pandas.concat([df1, df2, df3, df4, df5, df6, df7, df8],
                   ignore_index=True, axis=1)

df.columns = ["FVA_HS (Normal)",
              "FVA_HS (Fast)",
              "FVA_HS (Slow)",
              "FVA_HS (Carpet)",
              "GS_HS (Normal)",
              "GS_HS (Fast)",
              "GS_HS (Slow)",
              "GS_HS (Carpet)"]

del df1, df2, df3, df4, df5, df6, df7, df8

# create excel writer
# figname = (Filenames['patient_dir']+'/'+figname)
# plt.savefig(figname)
writer = pandas.ExcelWriter(Filenames['participant_num'] +
                            "_FVA_and_GS_Heel_Strikes.xlsx",
                            engine="xlsxwriter")

# write dataframe to excel sheet named 'marks'
df.to_excel(writer, sheet_name='Sheet1')

# # save the excel file
writer.save()
writer.close()
print('DataFrame is written successfully to Excel Sheet.')

# %% Compare HS detection methods
# Find closest matching pairs from SIMI and GS HS data
HeelStrike_SIMI = {
    "Normal": [],
    "Fast": [],
    "Slow":  [],
    "Carpet": []
}

HeelStrike_GS = {
    "Normal": [],
    "Fast": [],
    "Slow":  [],
    "Carpet": []
}

def _matching_HS(X, Y):

    z = []

    if len(X) < len(Y):
        for n in range(0, len(X)):
            num = min(Y, key=lambda x: abs(x-X[n]))
            if X[n] < num:
                z.append(np.array([X[n], num]))
    elif len(Y) < len(X):
        for n in range(0, len(Y)):
            num = min(X, key=lambda x: abs(x-Y[n]))
            if Y[n] < num:
                z.append(np.array([num, Y[n]]))

    del X, Y

    HeelStrike_SIMI = []
    HeelStrike_GS = []

    for n in range(0, len(z)):
        HeelStrike_SIMI.append(z[n][0])
        HeelStrike_GS.append(z[n][1])

    HeelStrike_SIMI = np.array(HeelStrike_SIMI)
    HeelStrike_GS = np.array(HeelStrike_GS)

    return HeelStrike_SIMI, HeelStrike_GS


# Matching Heel Strike Timing
[HeelStrike_SIMI['Normal'], HeelStrike_GS['Normal']] = _matching_HS(Batch_Outputs['FVA_vals_HS']['Normal'],
                                                          Batch_Outputs['GS_HS']['Normal'])
[HeelStrike_SIMI['Fast'], HeelStrike_GS['Fast']] = _matching_HS(Batch_Outputs['FVA_vals_HS']['Fast'],
                                                          Batch_Outputs['GS_HS']['Fast'])
[HeelStrike_SIMI['Slow'], HeelStrike_GS['Slow']] = _matching_HS(Batch_Outputs['FVA_vals_HS']['Slow'],
                                                          Batch_Outputs['GS_HS']['Slow'])
[HeelStrike_SIMI['Carpet'], HeelStrike_GS['Carpet']] = _matching_HS(Batch_Outputs['FVA_vals_HS']['Carpet'],
                                                          Batch_Outputs['GS_HS']['Carpet'])
# Save Matching HS to Spreadsheet

# Scatter Plot of Participant's HS from all trials
# plt.figure()
# plt.scatter(HeelStrike_GS['Normal'], HeelStrike_SIMI['Normal'])
# plt.scatter(HeelStrike_GS['Fast'], HeelStrike_SIMI['Fast'])
# plt.scatter(HeelStrike_GS['Slow'], HeelStrike_SIMI['Slow'])
# plt.scatter(HeelStrike_GS['Carpet'], HeelStrike_SIMI['Carpet'])
