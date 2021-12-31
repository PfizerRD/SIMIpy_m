# SIMI Motion Gait Metrics (SIMIpy-m)
# DMTI PfIRe Lab
# Authors: Visar Berki, Hao Zhang

# Library definitions:
# Installing packages directly from Spyder IPython Concole: !pip install scipy
# see: \AppData\Local\Programs\Spyder\pkgs
# create virtual environment with Anaconda if needed for installing packages


# Inputs:
# 1) Filepath for processed SIMI Motion marker data file (.txt format, delimiter='\t')
# 2) Filepath for processed GAITRite Walkway data file (.csv format, delimiter=',')
# 3) Filepath for processed GAITRite Walkway to SIMI SYNC file (.csv format, delimiter=',')
# Example:
# PN 12
# normal-paced walk
# filepath = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_SIMIMCS1_Normal_Walk_marker_processed_7_13_2021.txt"
# filepath_GS = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS.csv"
# filepath_GS_sync = r"C:\...\X9001262 - Data\10010012\X9001262_A_10010012_01_Normal_PKMAS_sync.csv"


def _metrics(filepath, filepath_GS, filepath_GS_sync, trial):

    import pandas
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy import signal
    from scipy.signal import argrelextrema
    from SIMIpy_m_PKMAS_sync import _get_PKMAS_sync
    from SIMIpy_m_Event_Metrics import _FVA_calc, _HMA_calc, _Heel_to_Heel
    from SIMIpy_m_metrics_SIMI_passes import _SIMI_passes_velocity

    plt.close("all")  # close all figures currently on screen

    # %% GAITRite PKMAS Sync

    # load GAITRite PKMAS Sync data, and time of sync signal (in seconds)
    # PKMAS_sync is the sync signal variable
    GS_PKMAS_sync, PKMAS_sync_t = _get_PKMAS_sync(filepath_GS_sync)

    # PKMAS_sync_t is the time that will be added to SIMIvars['time']['Time']
    # SIMI_data['Time'] = SIMI_data['Time'] + PKMAS_sync_t

    # %% Current analysis filenames (SIMI, GS, GS PKMAS Sync)
    current_trial = {}

    current_trial['GS_filename'] = os.path.basename(filepath_GS)
    current_trial['GS_sync_filename'] = os.path.basename(filepath_GS_sync)
    current_trial['SIMI_filename'] = os.path.basename(filepath)

    # %% Load SIMI File

    #   load SIMI Motion marker data from .txt data file, into dataframe
    SIMI_data = pandas.read_csv(filepath, delimiter='\t')
    #  sorted(SIMI_data)  generates a list of variable names

    def _droprows(df, num):
        # Drop first 2 rows (first 2 rows are sync signal. First row values are
        # all 0s, second row values are all 0.01) by selecting all rows
        # from 3rd row onwards

        df = df.drop(list(range(0, num))).reset_index(drop=True)

        return df

    # drop all rows of the dataframe above Time = 0
    num = np.where(SIMI_data['Time'] == 0)[0][1]
    SIMI_data = _droprows(SIMI_data, num=2)

    del num

    # Add PKMAS_sync_t to SIMI_data dataframe
    SIMI_data['Time'] = SIMI_data['Time'] + PKMAS_sync_t

    # %% SIMI Variables Dictionary
    #  Transform SIMI dataframe into a SIMI variables dictionary

    # Input: SIMI_data - format is 'DataFrame'
    # Output:
    # 1) SIMIvars - dictionary that stores the following keys (keys have format 'DataFrame'):
    # Heel_Left           - Heel, Left marker
    # Heel_Left_a         - Heel, Left marker Acceleration
    # Heel_Left_v         - Heel, Left marker Velocity
    # Heel_Right          - Heel, Right marker
    # Heel_Right_a        - Heel, Left marker Acceleration
    # Heel_Right_v        - Heel, Right marker Velocity
    # Spine               - Spine marker
    # Spine_a             - Spine marker Acceleration
    # Spine_v             - Spine marker Velocity
    # Thigh_Left          - Thigh, Left marker
    # Thigh_Left_a        - Thigh, Left marker Acceleration
    # Thigh_Left_v        - Thigh, Left marker Velocity
    # Thigh_Right         - Thigh, Right marker
    # Thigh_Right_a       - Thigh, Right marker acceleration
    # Thigh_Right_v       - Thigh, Right marker velocity
    # time                - time is unique here, it is stored as a series (rather than df)
    # Toe_Left            - Toe, Left marker
    # Toe_Left_a          - Toe, Left marker Acceleration
    # Toe_Left_v          - Toe, Left marker Velocity
    # Toe_Right           - Toe, Right marker
    # Toe_Right_a         - Toe, Right marker Acceleration
    # Toe_Right_v         - Toe, Right marker Velocity

    def _loadSIMIvars(df):
        # Dictionary of SIMI Motion Processed Data columns (.csv turned into dictionary with items)
        SIMIvars = {
            # time vector
            "time": df['Time'],
            # position vectors
            "Heel_Left": df[['heel left X', 'heel left Y', 'heel left Z', 'Time']],
            "Heel_Right": df[['heel right X', 'heel right Y', 'heel right Z', 'Time']],
            "Toe_Left": df[['foot tip left X', 'foot tip left Y', 'foot tip left Z', 'Time']],
            "Toe_Right": df[['foot tip right X', 'foot tip right Y', 'foot tip right Z', 'Time']],
            "Thigh_Left": df[['thigh left X', 'thigh left Y', 'thigh left Z', 'Time']],
            "Thigh_Right": df[['thigh right X', 'thigh right Y', 'thigh right Z', 'Time']],
            "Spine": df[['spine X', 'spine Y', 'spine Z', 'Time']],
            # velocity vectors
            "Heel_Left_v": df[['heel left v(X)', 'heel left v(Y)', 'heel left v(Z)', 'Time']],
            "Heel_Right_v": df[['heel right v(X)', 'heel right v(Y)', 'heel right v(Z)', 'Time']],
            "Toe_Left_v": df[['foot tip left v(X)', 'foot tip left v(Y)', 'foot tip left v(Z)', 'Time']],
            "Toe_Right_v": df[['foot tip right v(X)', 'foot tip right v(Y)', 'foot tip right v(Z)', 'Time']],
            "Thigh_Left_v": df[['thigh left v(X)', 'thigh left v(Y)', 'thigh left v(Z)', 'Time']],
            "Thigh_Right_v": df[['thigh right v(X)', 'thigh right v(Y)', 'thigh right v(Z)', 'Time']],
            "Spine_v": df[['spine v(X)', 'spine v(Y)', 'spine v(Z)', 'Time']],
            # acceleration vectors
            "Heel_Left_a": df[['heel left a(X)', 'heel left a(Y)', 'heel left a(Z)', 'Time']],
            "Heel_Right_a": df[['heel right a(X)', 'heel right a(Y)', 'heel right a(Z)', 'Time']],
            "Toe_Left_a": df[['foot tip left a(X)', 'foot tip left a(Y)', 'foot tip left a(Z)', 'Time']],
            "Toe_Right_a": df[['foot tip right a(X)', 'foot tip right a(Y)', 'foot tip right a(Z)', 'Time']],
            "Thigh_Left_a": df[['thigh left a(X)', 'thigh left a(Y)', 'thigh left a(Z)', 'Time']],
            "Thigh_Right_a": df[['thigh right a(X)', 'thigh right a(Y)', 'thigh right a(Z)', 'Time']],
            "Spine_a": df[['spine a(X)', 'spine a(Y)', 'spine a(Z)', 'Time']]
        }
        return SIMIvars

    SIMIvars_original = _loadSIMIvars(SIMI_data)
    SIMIvars = _loadSIMIvars(SIMI_data)

    # SIMIvars - Create SIMI variables dictionary from the original SIMI data with gaps
    SIMI_data_zeros = SIMI_data.fillna(value=0)
    SIMIvars_zeros = _loadSIMIvars(SIMI_data_zeros)
    del(SIMI_data_zeros)

    # %% Load GS File
    # load GAITRite System data from .csv file, into dataframe and dictionary

    # Distinguish between the different walkway passes
    # Collect GAITRite step timing events and assign them to variables

    # Input: filepath_GS - format is 'str'
    # Output:
    # 1) GS_vars - dictionary that stores the following keys (keys have format 'series'):
    # Absolute Step Length (cm.)
    # Cadence
    # First Contact (sec.)
    # Foot Angle (degrees)
    # Foot Center X Location (cm.)
    # Foot Center Y Location (cm.)
    # Foot Toe X Location (cm.)
    # Foot Toe Y Location (cm.)
    # Foot Heel X Location (cm.)
    # Foot Heel Y Location (cm.)
    # Gait Cycle Time (sec.)
    # Initial D. Support %
    # Initial D. Support (sec.)
    # Integ. Pressure (p x sec.)
    # Last Contact (sec.)
    # Single Support %
    # Single Support (sec.)
    # Stance %
    # Stance Time (sec.)
    # Step Length (cm.)
    # Step Time (sec.)
    # Stride Length (cm.)
    # Stride Time (sec.)
    # Stride Velocity (cm./sec.)
    # Stride Width (cm.)
    # Swing %
    # Swing Time (sec.)
    # Terminal D. Support %
    # Terminal D. Support (sec.)
    # Toe In/Out Angle (degrees)
    # Total D. Support %
    # Total D. Support (sec.)
    # Unnamed: 0
    # Velocity
    # Walk_Ratio

    GS_data = pandas.read_csv(filepath_GS, delimiter=',')
    skip = int(GS_data.iloc[:, 2][GS_data.iloc[:, 2] ==
               'Velocity (cm./sec.)'].index.to_numpy())
    GS_data = pandas.read_csv(filepath_GS, delimiter=',', skiprows=skip+1)

    GS_vars = GS_data.to_dict('series')

    GS_vars['Velocity'] = np.array(
        GS_vars['Velocity (cm./sec.)'][~np.isnan(GS_vars['Velocity (cm./sec.)'])])
    del GS_vars['Velocity (cm./sec.)']

    GS_vars['Cadence'] = np.array(
        GS_vars['Cadence (steps/min.)'][~np.isnan(GS_vars['Cadence (steps/min.)'])])
    del GS_vars['Cadence (steps/min.)']

    GS_vars['Walk_Ratio'] = np.array(
        GS_vars['Walk Ratio (cm./(steps/min.))'][~np.isnan(GS_vars['Walk Ratio (cm./(steps/min.))'])])
    del GS_vars['Walk Ratio (cm./(steps/min.))']

    GS_calc = {
        "Step_Events": []
    }
    GS_calc['Step_Events'] = GS_vars.pop('Unnamed: 1')

    # Drop the Nans at the beginning and end of the GS_calc['Step_Events'] array
    # careful doing this, becasue there are some NaN in the middle of some GS step data

    first_idx = GS_calc['Step_Events'].first_valid_index()
    last_idx = GS_calc['Step_Events'].last_valid_index()
    GS_calc['Step_Events'] = GS_calc['Step_Events'].loc[first_idx:last_idx]

    del first_idx, last_idx

    GS_calc['Step_Events'] = np.array(GS_calc['Step_Events'])

    # There are some header and blank cells above the GS_calc['Step_Events'] rows
    # skip those rows using this skip variable that gets assigned here
    skip = int(np.where(GS_vars['Unnamed: 0'] == '1')[0])
    GS_calc['Step_Events'] = np.delete(
        GS_calc['Step_Events'], range(0, skip-1), axis=0)
    del skip

    _str = GS_calc['Step_Events']

    GS_calc['Step_Events'] = pandas.DataFrame(GS_calc['Step_Events'])
    GS_calc['Step_Events'] = GS_calc['Step_Events'].set_axis(
        ['steps_idx'], axis=1, inplace=False)

    print("GS Steps: ", len(GS_calc['Step_Events']))

    GS_calc['Pass_1'] = []
    GS_calc['Pass_2'] = []
    GS_calc['Pass_3'] = []

    for n in range(0, len((GS_calc['Step_Events']))):
        if '1:' in _str[n]:
            GS_calc['Pass_1'].append(n)
        elif '2:' in _str[n]:
            GS_calc['Pass_2'].append(n)
        elif '3:' in _str[n]:
            GS_calc['Pass_3'].append(n)

    # %% GS Passes

    GS_calc['Pass_1'] = np.array(GS_calc['Pass_1'])
    GS_calc['Pass_2'] = np.array(GS_calc['Pass_2'])
    GS_calc['Pass_3'] = np.array(GS_calc['Pass_3'])

    # Drop the first two and last two steps from GS_calc Passes [2:-2]
    GS_calc['Pass_1'] = GS_calc['Pass_1']
    GS_calc['Pass_2'] = GS_calc['Pass_2']
    GS_calc['Pass_3'] = GS_calc['Pass_3']

    GS_calc['Pass_1_Left'] = []
    GS_calc['Pass_1_Right'] = []
    GS_calc['Pass_2_Left'] = []
    GS_calc['Pass_2_Right'] = []
    GS_calc['Pass_3_Left'] = []
    GS_calc['Pass_3_Right'] = []

    for n in range(0, len(GS_calc['Pass_1'])):
        if 'Left' in GS_calc['Step_Events'].steps_idx[GS_calc['Pass_1'][n]]:
            GS_calc['Pass_1_Left'].append(GS_calc['Pass_1'][n])
        elif 'Right' in GS_calc['Step_Events'].steps_idx[GS_calc['Pass_1'][n]]:
            GS_calc['Pass_1_Right'].append(GS_calc['Pass_1'][n])

    for n in range(0, len(GS_calc['Pass_2'])):
        if 'Left' in GS_calc['Step_Events'].steps_idx[GS_calc['Pass_2'][n]]:
            GS_calc['Pass_2_Left'].append(GS_calc['Pass_2'][n])
        elif 'Right' in GS_calc['Step_Events'].steps_idx[GS_calc['Pass_2'][n]]:
            GS_calc['Pass_2_Right'].append(GS_calc['Pass_2'][n])

    for n in range(0, len(GS_calc['Pass_3'])):
        if 'Left' in GS_calc['Step_Events'].steps_idx[GS_calc['Pass_3'][n]]:
            GS_calc['Pass_3_Left'].append(GS_calc['Pass_3'][n])
        elif 'Right' in GS_calc['Step_Events'].steps_idx[GS_calc['Pass_3'][n]]:
            GS_calc['Pass_3_Right'].append(GS_calc['Pass_3'][n])

    GS_calc['Left_Steps'] = np.concatenate([GS_calc['Pass_1_Left'],
                                            GS_calc['Pass_2_Left'],
                                            GS_calc['Pass_3_Left']])
    GS_calc['Right_Steps'] = np.concatenate([GS_calc['Pass_1_Right'],
                                             GS_calc['Pass_2_Right'],
                                             GS_calc['Pass_3_Right']])

    # Divide stride velocity by 100 to convert to m/s
    GS_calc['Midfoot_X'] = GS_vars['Foot Center X Location (cm.)'] / 100
    GS_calc['Midfoot_X'] = np.array(GS_calc['Midfoot_X'])
    GS_calc['Midfoot_X'] = np.delete(GS_calc['Midfoot_X'], range(0, 15), axis=0)

    GS_calc['First_Contact'] = GS_vars['First Contact (sec.)']
    GS_calc['First_Contact'] = np.array(GS_calc['First_Contact'])
    GS_calc['First_Contact'] = np.delete(
        GS_calc['First_Contact'], range(0, 15), axis=0)
    GS_calc['First_Contact'] = pandas.DataFrame(GS_calc['First_Contact'],
                                                columns=["Time"])
    GS_calc['First_Contact']['Side'] = [0] * len(GS_calc['First_Contact'])
    GS_calc['First_Contact']['Side'][GS_calc['Left_Steps']] = "left"
    GS_calc['First_Contact']['Side'][GS_calc['Right_Steps']] = "right"

    GS_calc['Last_Contact'] = GS_vars['Last Contact (sec.)']
    GS_calc['Last_Contact'] = np.array(GS_calc['Last_Contact'])
    GS_calc['Last_Contact'] = np.delete(
        GS_calc['Last_Contact'], range(0, 15), axis=0)

    # Divide stride velocity by 100 to convert to m/s
    GS_calc['Stride_Velocity'] = GS_vars['Stride Velocity (cm./sec.)'] / 100
    GS_calc['Stride_Velocity'] = np.array(GS_calc['Stride_Velocity'])
    GS_calc['Stride_Velocity'] = np.delete(
        GS_calc['Stride_Velocity'], range(0, 15), axis=0)
    GS_calc['Last_Contact'] = pandas.DataFrame(GS_calc['Last_Contact'],
                                               columns=["Time"])
    GS_calc['Last_Contact']['Side'] = [0] * len(GS_calc['Last_Contact'])
    GS_calc['Last_Contact']['Side'][GS_calc['Left_Steps']] = "left"
    GS_calc['Last_Contact']['Side'][GS_calc['Right_Steps']] = "right"
    # %% GS_calc - Mean Stride Velocity
    # GS_calc - Mean Stride Velocity (From GS Stride Velocity outputs)

    GS_calc['Stride_Velocity_mean'] = np.nanmean(GS_calc['Stride_Velocity'])

    # Stride Velocities of different passes. Data separated by pass number,
    # using GS "Stride Velocity" data channel

    GS_calc['Stride_Velocity_Pass_1'] = GS_calc['Stride_Velocity'][GS_calc['Pass_1']]
    GS_calc['Stride_Velocity_Pass_1_median'] = np.nanmedian(
        GS_calc['Stride_Velocity_Pass_1'])

    GS_calc['Stride_Velocity_Pass_2'] = GS_calc['Stride_Velocity'][GS_calc['Pass_2']]
    GS_calc['Stride_Velocity_Pass_2_median'] = np.nanmedian(
        GS_calc['Stride_Velocity_Pass_2'])

    GS_calc['Stride_Velocity_Pass_3'] = GS_calc['Stride_Velocity'][GS_calc['Pass_3']]
    GS_calc['Stride_Velocity_Pass_3_median'] = np.nanmedian(
        GS_calc['Stride_Velocity_Pass_3'])

    GS_calc['Stride_Velocity_mean'] = np.nanmean(
        np.concatenate(([GS_calc['Stride_Velocity_Pass_1'],
                         GS_calc['Stride_Velocity_Pass_2'],
                         GS_calc['Stride_Velocity_Pass_3']])))

    GS_calc['Stride_Velocity_Passes_mean'] = np.nanmean(
        [GS_calc['Stride_Velocity_Pass_1_median'],
         GS_calc['Stride_Velocity_Pass_2_median'],
         GS_calc['Stride_Velocity_Pass_3_median']])

    # Plot the GS Stride Velocity Data for each of the three passes

    # %% NaN Masks for Gaps
    # Create dictionary of NaN masks for cropping out missing data

    # time_vars will hold all channel masks that remove NaN values
    time_vars = {
        # Original number of samples (from processed SIMI Motion data)
        "Original_nSamples": len(SIMI_data)
    }
    # SIMIvars_no_gaps will hold all the SIMI Motion data with NaNs removed
    SIMIvars_no_gaps = {}

    # List of SIMI variable names
    # var_names = []
    # for key in sorted(SIMIvars):
    #     var_names.append(key)
    var_names = list(SIMIvars)

    # Append SIMIvars_no_gaps and time_vars dictionaries
    for key in var_names:
        if key == 'time':
            pass   # Do not do anything with original "time" channnel variable
        else:
            # Index of NaN values in data channel
            nan_array = np.isnan(SIMIvars[key])
            nanarray_sum = nan_array.sum()  # number of NaNs in data channel
            # Index of non-NaN values (good data mask) in data channel
            not_nan_array = ~nan_array
            # New data channel with only non-NaN values
            locals()[key] = SIMIvars[key][not_nan_array.iloc[:, 0]]
            # Index of non-NaN values (good data mask) in data channel
            time_vars[key + '_mask'] = not_nan_array.iloc[:, 0]
            del locals()[key]

    del [nan_array, nanarray_sum, not_nan_array, key]
    # SYNTAX:
    # SIMIvars_no_gaps['Heel_Left'].iloc[:,1]
    # SAME SYNTAX AS:
    # SIMIvars_no_gaps['Heel_Left']['heel left X']

    # SYNTAX for recreating gaps in gata:
    # a is an array with gap-filled values
    #  b is an array of True/False boolean values with 'False' where
    #  there were NaNs (gaps) in the original data, and 'True' where
    #  there was data present in that sample.

    # a[~b] = np.nan
    # Example: SIMIvars['Spine'][~time_vars['Spine_mask']] = np.nan
    #  Example 2: SIMIvars['Spine'][~time_vars['Spine_mask']] = np.nan

    # %% SIMI Data - Interpolated (Akima Spline)
    # SIMI data - Substitute NaN cells with interpolated values
    # Interpolation method is Akima spline
    # syntax: df_interpolated = df.interpolate(method='spline', order=5)
    # cubic spline method syntax: df_interpolated = df.interpolate(method='spline', order=5)

    SIMI_data = SIMI_data.interpolate(method='akima')

    data_columns = len(SIMI_data.iloc[1, :])

    # The spline interpolation only works between values of real numbers.
    # If the array starts with NaNs, those first NaNs will remain as NaNs
    # The following workaround sets all the NaNs at the start equal to the
    # value of the first real number in the array.

    # for n in range(1, data_columns-1):
    #     fill_idx = SIMI_data.iloc[:, n].first_valid_index()
    #     fill_value = SIMI_data.iloc[fill_idx, n]
    #     SIMI_data.iloc[:, n] = SIMI_data.iloc[:, n].fillna(value=fill_value)

    for n in range(1, data_columns-1):

        array_len = int(len(SIMI_data.iloc[:, n]))
        array_half_len = int(array_len/2)

        fill_idx = SIMI_data.iloc[:, n].first_valid_index()
        fill_value = SIMI_data.iloc[fill_idx, n]
        SIMI_data.iloc[0:array_half_len,
                       n] = SIMI_data.iloc[0:array_half_len, n].fillna(value=fill_value)

        fill_idx2 = SIMI_data.iloc[:, n].last_valid_index()
        fill_value2 = SIMI_data.iloc[fill_idx2, n]
        SIMI_data.iloc[array_half_len+1:array_len, n] = SIMI_data.iloc[array_half_len +
                                                                       1:array_len, n].fillna(value=fill_value2)

    del array_len, array_half_len

    # %% SIMI Variables Dictionary (Interpolated Data)
    SIMIvars = _loadSIMIvars(SIMI_data)

    # %% Apply PKMAS Sync to SIMIvars time

    # PKMAS_sync_t = time added to SIMIvars['time']
    SIMIvars['time_pre_sync'] = SIMIvars['time'] - PKMAS_sync_t
    # SIMIvars['time'] = SIMIvars['time'] + PKMAS_sync_t

    # %% Butterworth Filter definition

    # Butterworth filter will work for arrays with no NaN gaps.
    # For butterworth filter, the interpolated SIMI marker data is used

    def _butterworth_filter(Marker, time, _format, dim):
        # 4th Order Low-Pass Butterworth Filter, 7 Hz cutoff frequency
        # Syntax:
        # scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
        # scipy.signal.filtfilt(b, a, x, axis=- 1, padtype='odd', padlen=None, method='pad', irlen=None)

        if _format == 'df':
            Marker = Marker.to_numpy()    # Data Frame conversion to NumPy Array

        fs = 1/(time[1]-time[0])    # sampling frequency
        fc = 7                     # cutoff frequency
        b, a = signal.butter(4, fc/(fs), 'low', output='ba')
        # w, h = signal.freqz(b, a)
        if dim == 3:
            x = signal.filtfilt(b, a, Marker[:, 0])
            y = signal.filtfilt(b, a, Marker[:, 1])
            z = signal.filtfilt(b, a, Marker[:, 2])
            Marker_Filtered = np.array([x, y, z]).T
        elif dim == 1:
            x = signal.filtfilt(b, a, Marker)
            Marker_Filtered = np.array([x]).T

        return Marker_Filtered

    # Filter some variables (Butterworth: see fuction definition "_butterworth_filter")
    SIMIvars_Filtered = {
        # position vectors
        "Heel_Left": _butterworth_filter(SIMIvars['Heel_Left'], SIMIvars['time'], _format='df', dim=3),
        "Heel_Right": _butterworth_filter(SIMIvars['Heel_Right'], SIMIvars['time'], _format='df', dim=3),
        "Toe_Left": _butterworth_filter(SIMIvars['Toe_Left'], SIMIvars['time'], _format='df', dim=3),
        "Toe_Right": _butterworth_filter(SIMIvars['Toe_Right'], SIMIvars['time'], _format='df', dim=3),
        "Thigh_Left": _butterworth_filter(SIMIvars['Thigh_Left'], SIMIvars['time'], _format='df', dim=3),
        "Thigh_Right": _butterworth_filter(SIMIvars['Thigh_Right'], SIMIvars['time'], _format='df', dim=3),
        "Spine": _butterworth_filter(SIMIvars['Spine'], SIMIvars['time'], _format='df', dim=3),
        # velocity vectors
        "Heel_Left_v": _butterworth_filter(SIMIvars['Heel_Left_v'], SIMIvars['time'], _format='df', dim=3),
        "Heel_Right_v": _butterworth_filter(SIMIvars['Heel_Right_v'], SIMIvars['time'], _format='df', dim=3),
        "Toe_Left_v": _butterworth_filter(SIMIvars['Toe_Left_v'], SIMIvars['time'], _format='df', dim=3),
        "Toe_Right_v": _butterworth_filter(SIMIvars['Toe_Right_v'], SIMIvars['time'], _format='df', dim=3),
        "Thigh_Left_v": _butterworth_filter(SIMIvars['Thigh_Left_v'], SIMIvars['time'], _format='df', dim=3),
        "Thigh_Right_v": _butterworth_filter(SIMIvars['Thigh_Right_v'], SIMIvars['time'], _format='df', dim=3),
        "Spine_v": _butterworth_filter(SIMIvars['Spine_v'], SIMIvars['time'], _format='df', dim=3),
        # acceleration vectors
        "Heel_Left_a": _butterworth_filter(SIMIvars['Heel_Left_a'], SIMIvars['time'], _format='df', dim=3),
        "Heel_Right_a": _butterworth_filter(SIMIvars['Heel_Right_a'], SIMIvars['time'], _format='df', dim=3),
        "Toe_Left_a": _butterworth_filter(SIMIvars['Toe_Left_a'], SIMIvars['time'], _format='df', dim=3),
        "Toe_Right_a": _butterworth_filter(SIMIvars['Toe_Right_a'], SIMIvars['time'], _format='df', dim=3),
        "Thigh_Left_a": _butterworth_filter(SIMIvars['Thigh_Left_a'], SIMIvars['time'], _format='df', dim=3),
        "Thigh_Right_a": _butterworth_filter(SIMIvars['Thigh_Right_a'], SIMIvars['time'], _format='df', dim=3),
        "Spine_a": _butterworth_filter(SIMIvars['Spine_a'], SIMIvars['time'], _format='df', dim=3)
    }

    # %% Fill SIMIvars_no_gaps channels with original SIMI data, then apply NAN timing masks

    # List of SIMI variable names
    var_names = []
    for key in sorted(SIMIvars):
        var_names.append(key)

    for key in var_names:
        if key == 'time' or key == 'time_pre_sync':

            # Do not do anything with original "time_pre_sync" channnel variable
            # Do not do anything with the PKMAS sync "time" channnel variable
            pass   # Do not do anything with original "time" channnel variable
        else:
            # New data channel with only non-NaN values
            locals()[key] = SIMIvars[key][time_vars[key+'_mask']]
            SIMIvars_no_gaps[key] = eval(key)
            del locals()[key]

    # %% definition - Midpoint of Foot Calculator

    def _midpoint(p1, p2):

        x1 = p1[:, 0]
        y1 = p1[:, 1]
        z1 = p1[:, 2]

        x2 = p2[:, 0]
        y2 = p2[:, 1]
        z2 = p2[:, 2]

        Point = np.transpose(((x1 + x2)/2,
                              (y1 + y2)/2,
                              (z1 + z2)/2))
        return Point

    # %% Calculate Midpoint of Each Foot
    # Midpoint is halfway between heel and toe markers, for each foot.
    # To calculate, call the definition:  def _midpoint(p1, p2)

    # Foot midpoints (midpoint between each foot's heel and toe markers)
    SIMIvars_Filtered['Midpoint_Left_Foot'] = _midpoint(SIMIvars_Filtered['Heel_Left'],
                                                        SIMIvars_Filtered['Toe_Left'])

    SIMIvars_Filtered['Midpoint_Right_Foot'] = _midpoint(SIMIvars_Filtered['Heel_Right'],
                                                         SIMIvars_Filtered['Toe_Right'])

    # %% Midfoot Velocity - Differentiate midfoot position to calculate velocity

    # defining polynomial function
    var = np.poly1d([1, 0, 1])

    h = 0.01  # step size (1/Fs)
    SIMIvars_Filtered['Midpoint_Left_Foot_v'] = np.gradient(SIMIvars_Filtered['Midpoint_Left_Foot'],
                                                            edge_order=2, axis=0)/h

    SIMIvars_Filtered['Midpoint_Right_Foot_v'] = np.gradient(SIMIvars_Filtered['Midpoint_Right_Foot'],
                                                             edge_order=2, axis=0)/h

    time = np.array(SIMIvars['time'])

    # NaN Mask (time_vars['Heel_Left'] is Left Heel Mask)
    # SIMIvars_Filtered['Midpoint_Left_Foot_v'] = SIMIvars_Filtered['Midpoint_Left_Foot_v'][time_vars['Heel_Left_mask']]
    time_masked = np.array(SIMIvars['time'][time_vars['Heel_Left_mask']])

    SIMIvars_Filtered['Midpoint_Left_Foot_v'] = _butterworth_filter(SIMIvars_Filtered['Midpoint_Left_Foot_v'],
                                                                    time, _format='np', dim=3)
    SIMIvars_Filtered['Midpoint_Left_Foot_v'][~time_vars['Heel_Left_mask']] = np.NaN

    # SIMIvars_Filtered['Midpoint_Left_Foot_v'][~time_vars['Heel_Left_mask']] = np.NaN
    # SIMIvars_Filtered['Midpoint_Left_Foot_v'][~time_vars['Toe_Left_mask']] = np.NaN

    # SIMIvars_Filtered['Midpoint_Right_Foot_v'] = SIMIvars_Filtered['Midpoint_Right_Foot_v'][time_vars['Heel_Right_mask']]

    time_masked = np.array(SIMIvars['time'][time_vars['Heel_Right_mask']])

    SIMIvars_Filtered['Midpoint_Right_Foot_v'] = _butterworth_filter(SIMIvars_Filtered['Midpoint_Right_Foot_v'],
                                                                     time, _format='np', dim=3)
    SIMIvars_Filtered['Midpoint_Right_Foot_v'][~time_vars['Heel_Right_mask']] = np.NaN

    # SIMIvars_Filtered['Midpoint_Right_Foot_v'][~time_vars['Heel_Right_mask']] = np.NaN
    # SIMIvars_Filtered['Midpoint_Right_Foot_v'][~time_vars['Toe_Right_mask']] = np.NaN

    # %% Peaks Detection Definition

    def _peaks(signal, _format, _ord):

        if _format == 'np':
            df = pandas.DataFrame(signal, columns=['data'])
            # Find local maxima and minima
            n = 5  # number of points to be checked before and after
            # Find local peaks
            df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                                              order=n+_ord, mode='wrap')[0]]['data']
            df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                                              order=n+_ord, mode='wrap')[0]]['data']
        return df

    # %% FVA Algorithm - HS, TO Detection

    class _FVA_class:
        time = []
        HS_temp = []
        HS = []
        TO = []

    FVA_vars = _FVA_class()

    # Call definition for FVA peaks detection
    [FVA_Left_Foot, FVA_vars_left] = _FVA_calc(SIMIvars_original, SIMIvars_Filtered['Midpoint_Left_Foot_v'][:, 2],
                                               trial, time_vars['Heel_Left_mask'], _ord=1, side="left")
    [FVA_Right_Foot, FVA_vars_right] = _FVA_calc(SIMIvars_original, SIMIvars_Filtered['Midpoint_Right_Foot_v'][:, 2],
                                                 trial, time_vars['Heel_Right_mask'], _ord=1, side="right")

    FVA_vars.HS = FVA_vars_left['HS_vals'].append(FVA_vars_right['HS_vals'])
    FVA_vars.HS = FVA_vars.HS.sort_values(by="Time")
    FVA_vars.TO = FVA_vars_left['TO_vals'].append(FVA_vars_right['TO_vals'])
    FVA_vars.TO = FVA_vars.TO.sort_values(by="Time")

    del FVA_vars_left, FVA_vars_right

    # %% HMA Algorithm - HS, TO Detection
    # Hreljac, and Marshall - 2000
    # Algorithms to determine event timing during normal walking using kinematic data

    class HMA:
        time = []
        heel_accel_Z = []
        heel_accel_Z_peaks = []
        accel_horiz_peaks = []
        accel_horiz = []
        heel_jerk_Z = []
        heel_jerk_Z_peaks = []
        jerk_z_zeros = []
        jerk_z_zeros_time = []
        HS = []
        TO = []
        toe_accel_Z = []
        toe_jerk_horiz = []
        toe_jerk_peaks = []

    HMA_left = HMA()
    HMA_right = HMA()

    # Call definition for HMA peaks detection
    HMA_left_foot = _HMA_calc(SIMIvars['Heel_Left_a'],
                              SIMIvars['Toe_Left_a'],
                              time_vars['Heel_Left_mask'], HMA_left, h)

    HMA_right_foot = _HMA_calc(SIMIvars['Heel_Right_a'],
                               SIMIvars['Toe_Right_a'],
                               time_vars['Heel_Right_mask'], HMA_right, h)

    del HMA_left, HMA_right

    # %% Heel to Heel Algorithm - HS, TO Detection

    # Call definition for FVA peaks detection
    Heel_to_Heel = _Heel_to_Heel(SIMIvars_original, h)

    # %% Computed Gait Metrics from SIMI Motion (Markered Data)
    # This data will be calculated to match GaitRite walkway gait metrics
    # Velocity = Distance Traveled/time

    SIMI_metrics = {
        # time vector

        # Calculate Velocity from given SIMI Spine Marker Velocity vectors v(X), v(Y), and v(Z)
        "Spine_SIMI_v": np.array(np.sqrt(SIMIvars['Spine_v']['spine v(X)']**2 +
                                         SIMIvars['Spine_v']['spine v(Y)']**2 +
                                         SIMIvars['Spine_v']['spine v(Z)']**2))
    }

    # Remove NaN regions from marker data
    SIMI_metrics['Spine_SIMI_v_masked'] = SIMI_metrics['Spine_SIMI_v'][time_vars['Spine_mask']]

    # Mean Velocity of SIMI Spine Marker
    SIMI_metrics["Spine_SIMI_v_masked_mean"] = np.nanmean(
        SIMI_metrics['Spine_SIMI_v_masked'])

    # %% Gait Metrics - SIMI Velocity from Derivative of Instantaneous Position of Spine
    # Spine position data is derivated to produce 3 velocity vectors v(X), v(Y), v(Z)
    # Resultant velocity is calculated from velocity vectors

    # Velocity Derived from SIMI Spine Position Data ~~~~~~~~~~~~~~~~

    fs_step = round((SIMIvars['time'][1] - SIMIvars['time'][0]), 3)

    # Use gap filled data to compute instantaneous velocity, then crop out where gaps were
    SIMI_metrics['Spine_pos_deriv_velocity'] = np.diff(
        SIMIvars['Spine'].iloc[:, 0: 3], axis=0, prepend=1)/fs_step

    SIMI_metrics['Spine_pos_deriv_velocity'] = np.sqrt(SIMI_metrics['Spine_pos_deriv_velocity'][:, 0]**2 +
                                                       SIMI_metrics['Spine_pos_deriv_velocity'][:, 1]**2)

    # filter out sharp spikes
    SIMI_metrics['Spine_pos_deriv_velocity'] = signal.medfilt(
        SIMI_metrics['Spine_pos_deriv_velocity'], 5)

    # Crop out where the NaN position marker gaps were, from the calculated instantaneous velocity vectors
    # SIMI_metrics['Spine_pos_deriv_velocity2'] = SIMI_metrics['Spine_pos_deriv_velocity'][time_vars['Spine_mask']]
    # Place NaNs where original gaps were
    SIMI_metrics['Spine_pos_deriv_velocity'][~time_vars['Spine_mask']] = np.nan

    # Mean Velocity Calculated from 1st Derivative of SIMI Spine Position Data
    SIMI_metrics["Spine_pos_deriv_velocity_mean"] = np.nanmean(abs(
        SIMI_metrics['Spine_pos_deriv_velocity']))
    SIMI_metrics["Spine_pos_deriv_velocity_median"] = np.nanmedian(abs(
        SIMI_metrics['Spine_pos_deriv_velocity']))

    # %% Gait Metrics - SIMI Velocity from Pass times
    # One Pass = one pass through the GS platform
    # Three Passes per trial (X9001262 study)

    SIMI_metrics = _SIMI_passes_velocity(SIMIvars, SIMI_metrics)

    return(SIMI_metrics, GS_calc, GS_PKMAS_sync, FVA_vars)


# %% Saving a Figure

# # Save the figure
# figname = ('GS_SIMI_PN%s_Trial_%s'
#            % (Filenames['participant_num'], current_trial['GS_filename'][20:-4]))
# figname = (Filenames['patient_dir']+'/'+figname)
# plt.savefig(figname)

# del figname

# %% Saving variables, workspace to file

# ktk.save('filename.ktk.zip', variable)

# %% PLOTS

# _plot_GS_stride_Velocities()
# _plot_FVA()
# _plot_HMA()
# _plot_HHD()
