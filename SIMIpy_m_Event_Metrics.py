# SIMI Motion Gait Metrics (SIMIpy-m)
# DMTI PfIRe Lab
# Authors: Visar Berki, Hao Zhang

import pandas
import numpy as np
from scipy import signal
from scipy.signal import argrelextrema


# %% Butterworth filter definition


def _butterworth_filter(Marker, time, format, dim):
    # 4th Order Low-Pass Butterworth Filter, 7 Hz cutoff frequency
    # Syntax:
    # scipy.signal.butter(N, Wn, btype='low', analog=False, output='ba', fs=None)
    # scipy.signal.filtfilt(b, a, x, axis=- 1, padtype='odd', padlen=None, method='pad', irlen=None)

    if format == 'df':
        Marker = Marker.to_numpy()    # Data Frame conversion to NumPy Array

    fs = 1/(time[1]-time[0])    # sampling frequency
    fc = 7                      # cutoff frequency
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


# %% Foot Velocity Algorithm (FVA)

def _FVA_calc(variables, signal, trial, time_mask, ord):

    df = pandas.DataFrame(signal, columns=['data'])
    # Find local maxima and minima
    n = 5  # number of points to be checked before and after
    # Find local peaks
    df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                                      order=n+ord, mode='wrap')[0]]['data']
    df['TO'] = df['max'][df['max'] > 0.35]

    df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                                      order=n+ord, mode='wrap')[0]]['data']

    maxvals = df['TO'][~np.isnan(df['TO'])]
    minvals = df['min'][~np.isnan(df['min'])]
    df['HS'] = np.nan

    # the following for loop will pick out the last valley before each TO,
    # and record that valley as a HS event
    for n in range(0, len(maxvals)):
        idx = maxvals.index[n]

        # Make sure there is a valley before the first peak, then proceed
        if maxvals.index[n] > minvals.index[n]:
            minvals_temp = minvals[minvals.index <= idx]
            df['HS'][minvals_temp.index[-1]] = minvals_temp.iloc[-1]

    # Select only valleys close to zero (to match to HS)
    # df['min'] = df['min'][abs(df['min']) < 0.1]

    time_masked = variables['time'][time_mask]

    df['time'] = variables['time']

    FVA_vars = df.to_dict('series')
    FVA_vars['HS_vals'] = FVA_vars['HS'][~np.isnan(FVA_vars['HS'])]
    FVA_vars['HS_vals'] = variables['time'][FVA_vars['HS_vals'].index]

    FVA_vars['TO_vals'] = FVA_vars['TO'][~np.isnan(FVA_vars['TO'])]
    FVA_vars['TO_vals'] = variables['time'][FVA_vars['TO_vals'].index]

    return df, FVA_vars


# %% Hreljac-Marshall Algorithm (HMA)

# Dictionary Definition
def _HMA_calc(heel_accel, toe_accel, time_mask_heel, HMA, h):

    def _peaks_HMA(signal, format, ord):

        if format == 'np':
            df = pandas.DataFrame(signal, columns=['data'])
            # Find local maxima and minima
            # ord is number of points to be checked before and after
            # Find local peaks
            df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                                              order=ord, mode='clip')[0]]['data']
            df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                                              order=ord, mode='clip')[0]]['data']
            df['max'] = df['max'][df['max'] > 0.5]

        return df

    def _accel_jerk(accel, h):
        jerk = np.gradient(accel, edge_order=2, axis=0)/h
        return jerk

    HMA.time_heel = np.array(heel_accel.iloc[:, 3])
    HMA.time_toe = np.array(toe_accel.iloc[:, 3])
    # z acceleration for HMA
    HMA.heel_accel_Z = heel_accel.iloc[:, 2]
    HMA.toe_accel_Z = toe_accel.iloc[:, 2]

    # Butterworth Filter to smooth acceleration data
    # Syntax: _butterworth_filter(Marker, time, format, dim)
    HMA.heel_accel_Z = _butterworth_filter(HMA.heel_accel_Z, HMA.time_heel, format='np', dim=1)

    HMA.toe_accel_horiz = np.sqrt(toe_accel.iloc[:, 0]**2 +
                                  toe_accel.iloc[:, 1]**2)
    # HMA.toe_accel_horiz = toe_accel.iloc[:, 1]     # y acceleration (y-axis is along length of GS walkway)
    HMA.toe_accel_horiz = _butterworth_filter(HMA.toe_accel_horiz, HMA.time_toe, format='np', dim=1)

    # Vertical Jerk of Heel
    HMA.heel_jerk_Z = _accel_jerk(HMA.heel_accel_Z, h)

    # Horizontal Jerk of Toe
    HMA.toe_jerk_horiz = _accel_jerk(HMA.toe_accel_horiz, h)

    # Place NaNs where there were marker gaps in the original data (uses heel marker gaps)
    HMA.heel_accel_Z[~time_mask_heel] = np.NaN
    HMA.toe_accel_Z[~time_mask_heel] = np.NaN
    HMA.toe_accel_horiz[~time_mask_heel] = np.NaN
    HMA.heel_jerk_Z[~time_mask_heel] = np.NaN
    HMA.toe_jerk_horiz[~time_mask_heel] = np.NaN

    # HMA HS events (peaks detection)
    HMA.heel_accel_Z_peaks = _peaks_HMA(np.array(HMA.heel_accel_Z), format='np', ord=20)
    maxvals = HMA.heel_accel_Z_peaks['max'][~np.isnan(HMA.heel_accel_Z_peaks['max'])]
    HMA.HS = maxvals
    HMA.HS.time = HMA.time_heel[HMA.HS.index]

    del maxvals

    # HMA TO events (peaks detection)
    HMA.accel_horiz_peaks = _peaks_HMA(np.array(HMA.accel_horiz), format='np', ord=60)
    maxvals = HMA.accel_horiz_peaks['max'][~np.isnan(HMA.accel_horiz_peaks['max'])]
    HMA.TO = maxvals
    HMA.TO.time = HMA.time_toe[HMA.TO.index]

    # Heel jerk (vertical) (peaks detection)
    HMA.heel_jerk_Z_peaks = _peaks_HMA(np.array(HMA.heel_jerk_Z), format='np', ord=20)
    maxvals = HMA.heel_jerk_Z_peaks['max'][~np.isnan(HMA.heel_jerk_Z_peaks['max'])]
    HMA.heel_jerk_Z_peaks = maxvals
    HMA.heel_jerk_Z_peaks = HMA.time_heel[HMA.heel_jerk_Z_peaks.index]

    # # (Testing) - Modified HMA - TO event = peak of vertical jerk of toe marker
    # HMA TO events
    # HMA.toe_jerk_peaks = _peaks_HMA(np.array(HMA.toe_jerk_Z), format='np', ord=1)
    # maxvals = HMA.toe_jerk_peaks['max'][~np.isnan(HMA.toe_jerk_peaks['max'])]
    # HMA.TO = maxvals
    # HMA.TO.time = HMA.time_toe[HMA.TO.index]

    # HMA.jerk_z_zeros = HMA.jerk_Z[np.round(HMA.jerk_Z, 0) == 0]
    # HMA.jerk_z_zeros_time =  HMA.time_heel[np.where(HMA.jerk_Z == HMA.jerk_z_zeros)[0]]

    return HMA

# %% Heel to Heel Distance Algorithm


def _Heel_to_Heel(variables, h):

    def _peaks_Heel_to_Heel(signal, format, ord):

        if format == 'np':
            df = pandas.DataFrame(signal, columns=['data'])
            # Find local maxima and minima
            # ord is number of points to be checked before and after
            # Find local peaks
            df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                                              order=ord, mode='clip')[0]]['data']
            df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                                              order=ord, mode='clip')[0]]['data']
            df['max'] = df['max'][df['max'] > 5]

        return df

    heel_to_heel_dist = abs(np.array(variables['Heel_Left'].iloc[:, 0:3]) -
                            np.array(variables['Heel_Right'].iloc[:, 0:3]))
    return heel_to_heel_dist
