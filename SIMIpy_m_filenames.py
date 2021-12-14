# SIMI Motion Gait Metrics (SIMIpy-m)
# DMTI PfIRe Lab
# Authors: Visar Berki, Hao Zhang

# Library definitions:
# Installing packages directly from Spyder IPython Concole: !pip install datatest
# see: \AppData\Local\Programs\Spyder\pkgs
# create virtual environment with Anaconda if needed for installing packages

from tkinter import Tk, filedialog
import os
import sys

if 'Filenames' in locals():
    del Filenames

# Inputs:
# 1) Filepath for processed SIMI Motion marker data file (.txt format, delimiter='\t')
# 2) Filepath for processed GAITRite Walkway data file (.csv format, delimiter=',')
# Example:
# PN 12
# normal-paced walk
# filepath = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_SIMIMCS1_Normal_Walk_marker_processed_7_13_2021.txt"
# filepath_GS = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS.csv"

# Outputs:
# 1) Filenames - dictionary that stores the following keys (keys have format 'list' and 'str'):
# Filepair_Carpet_v1       - SIMI and GS files from visit day 1 (could be blank, one, or both files here)
# Filepair_Carpet_v2       - SIMI and GS files from visit day 2 (could be blank, one, or both files here)
# Filepair_Fast_v1         - "  "  for day 1 of fast gait trials
# Filepair_Fast_v2         - "  "  for day 2 of fast gait trials
# Filepair_Normal_v1       - "  "  for day 1 of normal pace gait trials
# Filepair_Normal_v2       - "  "  for day 2 of normal gait trials
# Filepair_Slow_v1         - "  "  for day 1 of slow gait trials
# Filepair_Slow_v2         - "  "  for day 2 of slow gait trials
# files                   - all files loaded in pre-processing
# files_GS                - list of GS filepaths
# files_SIMI              - list of SIMI filepaths
# participant_dir             - participant data directory
# participant_num             - participant number identifier (8 digits)
# trial_types             - types of gait trials (normal, fast, slow, carpet)

# To run and export variables, use this syntax:
    #       from SIMIpy_m_11_06_21 import SIMI_metrics

# % Choose participant Directory
# The available processed files will be registered as strings:
# Filenames['files_SIMI'], Filenames['files_GS']   -- (Python lists)

sys.path.append(r'C:\Users\berkiv\OneDrive - Pfizer\Documents\My Research Topics, Work Plan\X9001262 - Data')
sys.path.append(r'C:\Users\berkiv\OneDrive - Pfizer\Documents\GitHub\VisarPf')

Filenames = {}

# pointing Filenames['participant_dir'] to Tk() to use it as Tk() in program.
Filenames['participant_dir'] = Tk()
Filenames['participant_dir'].withdraw()  # Hides small tkinter window.

# Opened windows will be active. above all windows despite of selection.
Filenames['participant_dir'].attributes('-topmost', True)
path = filedialog.askdirectory()  # Returns opened path as str.
print(path)

Filenames['files'] = []
for Filenames['participant_dir'], dirs, files in os.walk(path):
    for file in files:
        if(file.endswith(".txt")) or file.endswith(".csv"):
            Filenames['files'].append(
                os.path.join(Filenames['participant_dir'], file))

# SIMI Motion filepaths:
Filenames['files_SIMI'] = []
for Filenames['participant_dir'], dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".txt"):
            Filenames['files_SIMI'].append(
                os.path.join(Filenames['participant_dir'], file))

Filenames['files_GS'] = []
# GAITRite System (GS) filepaths:
Filenames['files_GS'] = []
for Filenames['participant_dir'], dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):
            Filenames['files_GS'].append(
                os.path.join(Filenames['participant_dir'], file))

Filenames['participant_num'] = Filenames['participant_dir'][-8:]

del(file, files, dirs, path)


# % Pair each SIMI trial with its corresponding simultaneously collected GS trial
# if there is no SIMI-GS pair of processed files, there will be no comparison

Filenames['trial_types'] = ['Normal', 'Fast', 'Slow', 'Carpet']

Filepair_Normal = []
Filepair_Fast = []
Filepair_Slow = []
Filepair_Carpet = []


for n in range(0, len(Filenames['files'])):

    index = Filenames['files'][n].find('Normal')
    if index != -1:
        Filepair_Normal.append(Filenames['files'][n])

    index = Filenames['files'][n].find('Fast')
    if index != -1:
        Filepair_Fast.append(Filenames['files'][n])

    index = Filenames['files'][n].find('Slow')
    if index != -1:
        Filepair_Slow.append(Filenames['files'][n])

    index = Filenames['files'][n].find('Carpet')
    if index != -1:
        Filepair_Carpet.append(Filenames['files'][n])


# % Differentiate files during visit 1, from files during visit 2

Filepair_Normal_v1 = []
Filepair_Normal_v2 = []
Filepair_Fast_v1 = []
Filepair_Fast_v2 = []
Filepair_Slow_v1 = []
Filepair_Slow_v2 = []
Filepair_Carpet_v1 = []
Filepair_Carpet_v2 = []

def _visit_day_(Filenames, Filepair, Filepair_v1, Filepair_v2):
    # Check whether the pair of files are from the same participant visit day (data collection visit)
    for n in range(0, len(Filepair)):
        index = Filepair[n].find(Filenames['participant_num']+'_01')
        if index != -1:
            Filepair_v1.append(Filepair[n])
        else:
            Filepair_v2.append(Filepair[n])

    return Filepair_v1, Filepair_v2

Filenames['Filepair_Normal_v1'], Filenames['Filepair_Normal_v2'] = _visit_day_(
    Filenames, Filepair_Normal, Filepair_Normal_v1, Filepair_Normal_v2)
Filenames['Filepair_Fast_v1'], Filenames['Filepair_Fast_v2'] = _visit_day_(
    Filenames, Filepair_Fast, Filepair_Fast_v1, Filepair_Fast_v2)
Filenames['Filepair_Slow_v1'], Filenames['Filepair_Slow_v2'] = _visit_day_(
    Filenames, Filepair_Slow, Filepair_Slow_v1, Filepair_Slow_v2)
Filenames['Filepair_Carpet_v1'], Filenames['Filepair_Carpet_v2'] = _visit_day_(
    Filenames, Filepair_Carpet, Filepair_Carpet_v1, Filepair_Carpet_v2)

del (Filepair_Normal, Filepair_Fast, Filepair_Slow, Filepair_Carpet, index, n)
del(Filepair_Normal_v1, Filepair_Normal_v2, Filepair_Fast_v1, Filepair_Fast_v2,
    Filepair_Slow_v1, Filepair_Slow_v2, Filepair_Carpet_v1, Filepair_Carpet_v2)