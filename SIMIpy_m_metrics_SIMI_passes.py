import numpy as np
import pandas
from scipy.signal import argrelextrema

# %% Peaks Detection Definition


def __peaks(signal, format, ord):

    if format == 'np':
        df = pandas.DataFrame(signal, columns=['data'])
        # Find local maxima and minima
        n = 5  # number of points to be checked before and after
        # Find local peaks
        df['min'] = df.iloc[argrelextrema(df.data.values, np.less_equal,
                                          order=n+ord, mode='wrap')[0]]['data']
        df['max'] = df.iloc[argrelextrema(df.data.values, np.greater_equal,
                                          order=n+ord, mode='wrap')[0]]['data']

    return df


# %% Gait Metrics - SIMI Velocity from Pass times
# One Pass = one pass through the GS platform
# Three Passes per trial (X9001262 study)

def _SIMI_passes_velocity(SIMIvars, SIMI_metrics):
    arr = np.array(SIMIvars['Spine'])
    arr[:, 0:3][arr[:, 1] <= -1.5] = 0
    arr[:, 0:3][arr[:, 1] >= 1.5] = 0

    t_arr = arr[:, 3]
    df = __peaks(arr[:, 1], format='np', ord=10)
    # remove all the zero valued peaks originating from gaps in marker data
    df['max'] = df['max'][df['max'] != 0]
    # remove all the zero valued valleys originating from gaps in marker data
    df['min'] = df['min'][df['min'] != 0]

    # plt.figure()
    # plt.plot(t_arr, arr[:, 1])

    # Peak positions (in y direction), to determine SIMI walkway passes
    pandas.options.mode.chained_assignment = None  # default='warn'
    min_pos = df[df['min'].notnull()]
    min_pos['time'] = t_arr[min_pos.index]
    max_pos = df[df['max'].notnull()]
    max_pos['time'] = t_arr[max_pos.index]
    pos = pandas.concat([min_pos, max_pos]).sort_index()

    SIMI_metrics['Pass_Start'] = [pos['time'].iloc[0],
                                  pos['time'].iloc[2],
                                  pos['time'].iloc[4]]
    SIMI_metrics['Pass_End'] = [pos['time'].iloc[1],
                                pos['time'].iloc[3],
                                pos['time'].iloc[5]]

    # Calculate time to travel from start of pass to end of pass
    # Pass Duration
    SIMI_metrics['Pass_duration'] = []
    SIMI_metrics['Pass_duration'].append(
        SIMI_metrics['Pass_End'][0] - SIMI_metrics['Pass_Start'][0])
    SIMI_metrics['Pass_duration'].append(
        SIMI_metrics['Pass_End'][1] - SIMI_metrics['Pass_Start'][1])
    SIMI_metrics['Pass_duration'].append(
        SIMI_metrics['Pass_End'][2] - SIMI_metrics['Pass_Start'][2])

    # Pass Length
    SIMI_metrics['Pass_length'] = []
    SIMI_metrics['Pass_length'].append(
        np.sqrt((pos['data'].iloc[0] - pos['data'].iloc[1])**2))
    SIMI_metrics['Pass_length'].append(
        np.sqrt((pos['data'].iloc[2] - pos['data'].iloc[3])**2))
    SIMI_metrics['Pass_length'].append(
        np.sqrt((pos['data'].iloc[4] - pos['data'].iloc[5])**2))

    # Pass Velocity
    SIMI_metrics['Velocity_Passes'] = []
    SIMI_metrics['Velocity_Passes'].append(
        abs(SIMI_metrics['Pass_length'][0] / SIMI_metrics['Pass_duration'][0]))
    SIMI_metrics['Velocity_Passes'].append(
        abs(SIMI_metrics['Pass_length'][1] / SIMI_metrics['Pass_duration'][1]))
    SIMI_metrics['Velocity_Passes'].append(
        abs(SIMI_metrics['Pass_length'][2] / SIMI_metrics['Pass_duration'][2]))

    SIMI_metrics['Velocity_Passes_mean'] = np.nanmean(SIMI_metrics['Velocity_Passes'])

    return SIMI_metrics
