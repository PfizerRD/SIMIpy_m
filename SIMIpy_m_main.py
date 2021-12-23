# SIMI Motion Gait Metrics (SIMIpy-m)
# DMTI PfIRe Lab
# Authors: Visar Berki, Hao Zhang

# Definitions

import os
import pandas
import numpy as np
# from SIMIpy_m_filenames import Filenames
from SIMIpy_m_processing_filepair import _filepair_chooser
import SIMIpy_m_filenames
import importlib
import matplotlib as plt
importlib.reload(SIMIpy_m_filenames)

path = os.getcwd()
os.chdir(path)

Filenames = SIMIpy_m_filenames.Filenames
Processing_filepairs = _filepair_chooser(Filenames)

# %% Save all processed file pairs for one participant into "Batch_Outputs" dictionary

Batch_Outputs = {
    "Filenames": [],
    "GS_calc": [],
    "GS_HS": [],
    "GS_TO": [],
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
    Batch_Outputs['GS_HS'][trial] = GS_calc['First_Contact']
    Batch_Outputs['GS_TO'][trial] = GS_calc['Last_Contact']
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
df = pandas.concat([df1['Time'], df1['Side'],
                    df2['Time'], df2['Side'],
                    df3['Time'], df3['Side'],
                    df4['Time'], df4['Side'],
                    df5['Time'], df5['Side'],
                    df6['Time'], df6['Side'],
                    df7['Time'], df7['Side'],
                    df8['Time'], df8['Side']],
                    ignore_index=True, axis=1)

df.columns = ["FVA_HS (Normal)", "Side (Normal)",
              "FVA_HS (Fast)", "Side (Fast)",
              "FVA_HS (Slow)", "Side (Slow)",
              "FVA_HS (Carpet)", "Side (Carpet)",
              "GS_HS (Normal)", "Side (Normal)",
              "GS_HS (Fast)", "Side (Fast)",
              "GS_HS (Slow)", "Side (Slow)",
              "GS_HS (Carpet)", "Side (Carpet)"]

del df1, df2, df3, df4, df5, df6, df7, df8

# create excel writer

# writer = pandas.ExcelWriter(Filenames['participant_num'] +
#                             "_FVA_and_GS_Heel_Strikes.xlsx",
#                             engine="xlsxwriter")

# write dataframe to excel sheet named 'marks'
# df.to_excel(writer, sheet_name='Sheet1')

# # save the excel file
# writer.save()
# writer.close()
# print('Heel Strikes DataFrame is written successfully to Excel Sheet.')

# %% EVENT METRICS SPREADSHEET (TO)

df1 = pandas.DataFrame(Batch_Outputs['FVA_vals_TO']['Normal'])
df2 = pandas.DataFrame(Batch_Outputs['FVA_vals_TO']['Fast'])
df3 = pandas.DataFrame(Batch_Outputs['FVA_vals_TO']['Slow'])
df4 = pandas.DataFrame(Batch_Outputs['FVA_vals_TO']['Carpet'])

df5 = pandas.DataFrame(Batch_Outputs['GS_TO']['Normal'])
df6 = pandas.DataFrame(Batch_Outputs['GS_TO']['Fast'])
df7 = pandas.DataFrame(Batch_Outputs['GS_TO']['Slow'])
df8 = pandas.DataFrame(Batch_Outputs['GS_TO']['Carpet'])
df = pandas.concat([df1['Time'], df1['Side'],
                    df2['Time'], df2['Side'],
                    df3['Time'], df3['Side'],
                    df4['Time'], df4['Side'],
                    df5['Time'], df5['Side'],
                    df6['Time'], df6['Side'],
                    df7['Time'], df7['Side'],
                    df8['Time'], df8['Side']],
                   ignore_index=True, axis=1)

df.columns = ["FVA_TO (Normal)", "Side (Normal)",
              "FVA_TO (Fast)", "Side (Fast)",
              "FVA_TO (Slow)", "Side (Slow)",
              "FVA_TO (Carpet)", "Side (Carpet)",
              "GS_TO (Normal)", "Side (Normal)",
              "GS_TO (Fast)", "Side (Fast)",
              "GS_TO (Slow)", "Side (Slow)",
              "GS_TO (Carpet)", "Side (Carpet)"]

del df1, df2, df3, df4, df5, df6, df7, df8
# create excel writer

# writer = pandas.ExcelWriter(Filenames['participant_num'] +
#                             "_FVA_and_GS_Toe_Off.xlsx",
#                             engine="xlsxwriter")

# write dataframe to excel sheet named 'marks'
# df.to_excel(writer, sheet_name='Sheet1')

# # save the excel file
# writer.save()
# writer.close()
# print('Toe Off DataFrame is written successfully to Excel Sheet.')

# %% Compare HS detection methods
# Find closest matching pairs from SIMI and GS HS data
HeelStrike_SIMI = {
    "Normal": [],
    "Fast": [],
    "Slow":  [],
    "Carpet": [],
    "All": []
}

HeelStrike_GS = {
    "Normal": [],
    "Fast": [],
    "Slow":  [],
    "Carpet": [],
    "All": []
}


def _matching_HS(X, Y):

    SIMIvar = np.array(X)
    GSvar = np.array(Y)
    X = np.array(X)[:, 0]
    Y = np.array(Y)[:, 0]

    z = []

    if len(X) < len(Y):
        for n in range(0, len(X)):
            num = min(Y, key=lambda x: abs(x-X[n]))
            z.append([SIMIvar[n], [num, GSvar[n,1]]])
    elif len(Y) < len(X):
        for n in range(0, len(Y)):
            num = min(X, key=lambda x: abs(x-Y[n]))
            z.append([[num, GSvar[n,1]], GSvar[n]])

    del X, Y, SIMIvar, GSvar


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

HeelStrike_SIMI['All'] = np.concatenate([HeelStrike_SIMI['Normal'],
                                        HeelStrike_SIMI['Fast'],
                                        HeelStrike_SIMI['Slow'],
                                        HeelStrike_SIMI['Carpet']])

HeelStrike_GS['All'] = np.concatenate([HeelStrike_GS['Normal'],
                                      HeelStrike_GS['Fast'],
                                      HeelStrike_GS['Slow'],
                                      HeelStrike_GS['Carpet']])

# Save Matching HS to Spreadsheet

# Scatter Plot of Participant's HS from all trials
# plt.figure()
# plt.scatter(HeelStrike_GS['Normal'], HeelStrike_SIMI['Normal'])
# plt.scatter(HeelStrike_GS['Fast'], HeelStrike_SIMI['Fast'])
# plt.scatter(HeelStrike_GS['Slow'], HeelStrike_SIMI['Slow'])
# plt.scatter(HeelStrike_GS['Carpet'], HeelStrike_SIMI['Carpet'])

# %% Compare TO detection methods
# Find closest matching pairs from SIMI and GS TO data

ToeOff_SIMI = {
    "Normal": [],
    "Fast": [],
    "Slow":  [],
    "Carpet": [],
    "All": []
}

ToeOff_GS = {
    "Normal": [],
    "Fast": [],
    "Slow":  [],
    "Carpet": [],
    "All": []
}


def _matching_TO(X, Y):

    # X = np.array(X)
    # Y = np.array(Y)
    # z = []

    # if len(X) < len(Y):
    #     for n in range(0, len(X)):
    #         num = min(Y[:,0], key=lambda x: abs(x-X[:,0][n]))
    #         if X[:,0][n] < num:
    #             z.append([X[n], num])
    # elif len(Y) < len(X):
    #     for n in range(0, len(Y)):
    #         num = min(X[:,0], key=lambda x: abs(x-Y[:,0][n]))
    #         if Y[:,0][n] < num:
    #             z.append([num, Y[n]])

    # del X, Y

    SIMIvar = np.array(X)
    GSvar = np.array(Y)
    X = np.array(X)[:, 0]
    Y = np.array(Y)[:, 0]

    z = []

    if len(X) < len(Y):
        for n in range(0, len(X)):
            num = min(Y, key=lambda x: abs(x-X[n]))
            z.append([SIMIvar[n], [num, GSvar[n,1]]])
    elif len(Y) < len(X):
        for n in range(0, len(Y)):
            num = min(X, key=lambda x: abs(x-Y[n]))
            z.append([[num, GSvar[n,1]], GSvar[n]])

    del X, Y, SIMIvar, GSvar


    ToeOff_SIMI = []
    ToeOff_GS = []

    for n in range(0, len(z)):
        ToeOff_SIMI.append(z[n][0])
        ToeOff_GS.append(z[n][1])

    ToeOff_SIMI = np.array(ToeOff_SIMI)
    ToeOff_GS = np.array(ToeOff_GS)

    return ToeOff_SIMI, ToeOff_GS


# Matching Toe Off Timing
[ToeOff_SIMI['Normal'], ToeOff_GS['Normal']] = _matching_TO(Batch_Outputs['FVA_vals_TO']['Normal'],
                                                            Batch_Outputs['GS_TO']['Normal'])
[ToeOff_SIMI['Fast'], ToeOff_GS['Fast']] = _matching_TO(Batch_Outputs['FVA_vals_TO']['Fast'],
                                                        Batch_Outputs['GS_TO']['Fast'])
[ToeOff_SIMI['Slow'], ToeOff_GS['Slow']] = _matching_TO(Batch_Outputs['FVA_vals_TO']['Slow'],
                                                        Batch_Outputs['GS_TO']['Slow'])
[ToeOff_SIMI['Carpet'], ToeOff_GS['Carpet']] = _matching_TO(Batch_Outputs['FVA_vals_TO']['Carpet'],
                                                            Batch_Outputs['GS_TO']['Carpet'])

ToeOff_SIMI['All'] = np.concatenate([ToeOff_SIMI['Normal'],
                                     ToeOff_SIMI['Fast'],
                                     ToeOff_SIMI['Slow'],
                                     ToeOff_SIMI['Carpet']])

ToeOff_GS['All'] = np.concatenate([ToeOff_GS['Normal'],
                                   ToeOff_GS['Fast'],
                                   ToeOff_GS['Slow'],
                                   ToeOff_GS['Carpet']])

# %% Compile all FVA and GS HS and TO
# Results compiled into one dictionary that contains all participants and all trials

_mykeys = {"HeelStrike_SIMI_Normal": {},
           "HeelStrike_SIMI_Fast": {},
           "HeelStrike_SIMI_Slow": {},
           "HeelStrike_SIMI_Carpet": {},

           "ToeOff_SIMI_Normal": {},
           "ToeOff_SIMI_Fast": {},
           "ToeOff_SIMI_Slow": {},
           "ToeOff_SIMI_Carpet": {},

           "HeelStrike_GS_Normal": {},
           "HeelStrike_GS_Fast": {},
           "HeelStrike_GS_Slow": {},
           "HeelStrike_GS_Carpet": {},

           "ToeOff_GS_Normal": {},
           "ToeOff_GS_Fast": {},
           "ToeOff_GS_Slow": {},
           "ToeOff_GS_Carpet": {}
           }

if 'All_Participants_HS_TO' not in locals():
    All_Participants_HS_TO = {
        "PN10010002": _mykeys.copy(),
        "PN10010003": _mykeys.copy(),
        "PN10010004": _mykeys.copy(),
        "PN10010005": _mykeys.copy(),
        "PN10010006": _mykeys.copy(),
        "PN10010007": _mykeys.copy(),
        "PN10010008": _mykeys.copy(),
        "PN10010009": _mykeys.copy(),
        "PN10010010": _mykeys.copy(),
        "PN10010011": _mykeys.copy(),
        "PN10010012": _mykeys.copy(),
        "PN10010013": _mykeys.copy(),
        "PN10010014": _mykeys.copy(),
        "PN10010015": _mykeys.copy(),
        "PN10010016": _mykeys.copy(),
        "PN10010017": _mykeys.copy(),
        "PN10010018": _mykeys.copy(),
        "PN10010019": _mykeys.copy(),
        "PN10010020": _mykeys.copy(),
    }


z = "PN" + Filenames['participant_num']
print(z)

if 'All_Participants_HS_TO' in locals():
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_SIMI_Normal"] = HeelStrike_SIMI['Normal']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_SIMI_Fast"] = HeelStrike_SIMI['Fast']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_SIMI_Slow"] = HeelStrike_SIMI['Slow']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_SIMI_Carpet"] = HeelStrike_SIMI['Carpet']

    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_SIMI_Normal"] = ToeOff_SIMI['Normal']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_SIMI_Fast"] = ToeOff_SIMI['Fast']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_SIMI_Slow"] = ToeOff_SIMI['Slow']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_SIMI_Carpet"] = ToeOff_SIMI['Carpet']

    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_GS_Normal"] = HeelStrike_GS['Normal']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_GS_Fast"] = HeelStrike_GS['Fast']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_GS_Slow"] = HeelStrike_GS['Slow']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["HeelStrike_GS_Carpet"] = HeelStrike_GS['Carpet']

    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_GS_Normal"] = ToeOff_GS['Normal']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_GS_Fast"] = ToeOff_GS['Fast']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_GS_Slow"] = ToeOff_GS['Slow']
    All_Participants_HS_TO["PN" + Filenames['participant_num']]["ToeOff_GS_Carpet"] = ToeOff_GS['Carpet']

del z
