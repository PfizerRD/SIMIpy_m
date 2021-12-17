# %% These can run after processing SIMI and GS Metrics

import pandas
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.signal import argrelextrema
import math

# %% Plot the GS Stride Velocity Data for each of the three passes


def _plot_GS_stride_Velocities():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.show()
    plt.grid(color='0.95')

    plt.scatter(GS_calc['First_Contact_time'][GS_calc['Pass_1']], GS_calc['Stride_Velocity_Pass_1'],
                label='GS Stride_Velocity_Pass_1', s=75)
    plt.scatter(GS_calc['First_Contact_time'][GS_calc['Pass_2']], GS_calc['Stride_Velocity_Pass_2'],
                label='GS Stride_Velocity_Pass_2', s=75)
    plt.scatter(GS_calc['First_Contact_time'][GS_calc['Pass_3']], GS_calc['Stride_Velocity_Pass_3'],
                label='GS Stride_Velocity_Pass_3', s=75)
    plt.title('GS Stride Velocities for Three Walkway Passes', size=13)
    plt.xlabel('Time (s)')
    plt.ylabel('Stride Velocity v(Y) (cm/s)')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend()
    lgd = ax.legend(handles, labels, loc='lower center', ncol=3,
                    bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()


_plot_GS_stride_Velocities()

# %% Plot Original Variable, Filtered Variable, Filtered Variable with no gaps

# plt.figure()
# plt.scatter(SIMIvars['time'], SIMIvars_original['Spine']
#             ['spine Y'], label='Original Marker Data')
# plt.scatter(SIMIvars['time'], SIMIvars_Filtered['Spine']
#             [:, 1], label='Filtered Marker Data')
# plt.scatter(SIMIvars['time'][time_vars['Spine_mask']], SIMIvars_no_gaps['Spine']['spine Y'],
#             label='Gapless Marker Data')
# plt.title('Spine Position', size=13)
# plt.xlabel('time (s)')
# plt.ylabel('Position (y)')
# plt.legend(loc='lower left', borderaxespad=0.)
# plt.tight_layout()
# plt.show()

# %% Plot HMA Results


def _plot_HMA():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='0.95', linestyle='dashed')
    ax.xaxis.grid(color='0.95', linestyle='dashed')
    plt.show()

    # Plot Heel Vertical Acceleration (Left Foot)
    plt.plot(HMA_left_foot.time_heel, HMA_left_foot.heel_accel_Z, c='g',
             label='Vertical Acceleration (Left Heel Marker)')

    # # # Plot HS, Left Foot (Vertical Heel Accel Peaks) ~~~~~~~~~~~~~~~~~~~~~~~~
    # plt.scatter(HMA_left_foot.HS.time, [0] * len(HMA_left_foot.HS.time), c='g',
    #             s=75, label='HS, Z Accel Peaks (Left Foot)')
    plt.scatter(HMA_left_foot.HS.time, HMA_left_foot.HS, c='g', s=75,
                label='HS, Z Accel Peaks (Left Foot)')

    # # # Plot HS, Right Foot (Vertical Heel Accel Peaks) ~~~~~~~~~~~~~~~~~~~~~~~~
    # plt.plot(HMA_right_foot.time_heel, HMA_right_foot.heel_accel_Z, c='orange',
    #           label='Vertical Acceleration (Right Heel Marker)')

    # # # Plot HS, Right Foot (Vertical Heel Accel Peaks) ~~~~~~~~~~~~~~~~~~~~~~~~
    # plt.scatter(HMA_right_foot.HS.time, [0] * len(HMA_right_foot.HS.time), c='orange',
    #             s=75, label='HS, Z Accel Peaks (Right Foot)')
    # plt.scatter(HMA_right_foot.HS.time, HMA_right_foot.HS, c='orange', s=75,
    #             label='HS, Z Accel Peaks (Right Foot)')

    # # # Plot TO, Left Foot (Horizontal Toe Accel Peaks) ~~~~~~~~~~~~~~~~~~~~~~~~
    # plt.plot(HMA_left_foot.time_toe, HMA_left_foot.toe_accel_horiz,
    #           c='magenta', label='Horizontal Acceleration (Left Toe Marker)')
    # plt.scatter(HMA_left_foot.TO.time, [0] * len(HMA_left_foot.TO.time), c='magenta',
    #             s=75, label='TO, Horiz Accel Peaks (Left Foot)')
    # plt.scatter(HMA_left_foot.TO.time, HMA_left_foot.TO, c='magenta',
    #             s=75, label='TO, Horiz Accel Peaks (Left Foot)')

    # # # Plot TO, Right Foot (Horizontal Toe Accel Peaks) ~~~~~~~~~~~~~~~~~~~~~~~~
    # plt.plot(HMA_right_foot.time_toe, HMA_right_foot.toe_accel_horiz, c='magenta', label='Horizontal Acceleration (Left Toe Marker)')
    # plt.scatter(HMA_right_foot.TO.time, HMA_right_foot.TO, c='magenta', s=75,
    #             label='TO, Horiz Accel Peaks (Right Foot)')

    # # # Plot Horizontal Jerk of Left Toe Marker ~~~~~~~~~~~~~~~~~~~~~~~~
    # plt.plot(HMA_left_foot.time_toe, HMA_left_foot.toe_jerk_horiz, c='b',
    #           label='Horizontal Jerk (Left Toe Marker)')
    # # plt.scatter(HMA_left_foot.jerk_z_zeros_time, HMA_left_foot.jerk_z_zeros,
    # #             c='magenta', label='Vertical Jerk (Left Heel Marker)')
    # plt.scatter(HMA_left_foot.TO.time, HMA_left_foot.TO, s=75,
    #             c='magenta', label='Vertical Jerk (Left Heel Marker)')

    # # Plot Vertical Jerk of Left Heel Marker
    plt.plot(HMA_left_foot.time_heel, HMA_left_foot.heel_jerk_Z, c='m', label='Vertical Jerk (Left Heel Marker)')
    plt.scatter(HMA_left_foot.heel_jerk_Z_peaks, [0] * len(HMA_left_foot.heel_jerk_Z_peaks),
                c='magenta', label='Vertical Jerk (Left Heel Marker)')
    # plt.scatter(HMA_left_foot.jerk_z_zeros_time, HMA_left_foot.jerk_z_zeros, c='magenta', label='Vertical Jerk (Left Heel Marker)')

    # # Plot Vertical Jerk of Right Heel Marker
    # plt.plot(HMA_right_foot.time_heel, HMA_right_foot.heel_jerk_Z, c='b', label='Vertical Jerk (Right Heel Marker)')
    # plt.scatter(HMA_right_foot.jerk_z_zeros_time, HMA_right_foot.jerk_z_zeros, c='magenta', label='Vertical Jerk (Right Heel Marker)')

    # Scatter plot GS First Contact Times of different passes
    plt.title('HMA Algorithm - GS and SIMI Velocities \n Participant (#%s) \n Trial SIMI: (%s)'
              % (Filenames['participant_num'],
                 current_trial['SIMI_filename'][20:]), size=16)
    plt.scatter(GS_calc['First_Contact_time'], GS_calc['Stride_Velocity'],
                label='GS First Contact Times', s=75, c='gold')
    # Scatter plot GS Last Contact Times of different passes
    plt.scatter(GS_calc['Last_Contact_time'], GS_calc['Stride_Velocity'],
                label='GS Last Contact Times', s=75, c='black')

    handles, labels = ax.get_legend_handles_labels()
    plt.legend()
    lgd = ax.legend(handles, labels, loc='lower center', ncol=3,
                    bbox_to_anchor=(0.5, -0.1))
    plt.tight_layout()
    # ax.set_aspect(0.2)

    # plt.figure()


_plot_HMA()

# %% Plot results of FVA


def _plot_FVA():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.show()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='0.95', linestyle='dashed')
    ax.xaxis.grid(color='0.95', linestyle='dashed')

    # FVA Left Foot Peaks
    plt.scatter(FVA_Left_Foot['time'], FVA_Left_Foot['TO'],
                s=75, c='g', label='TO (FVA)')
    plt.scatter(FVA_Left_Foot['time'], FVA_Left_Foot['HS'],
                s=75, c='r', label='HS (FVA)')
    plt.plot(FVA_Left_Foot['time'], FVA_Left_Foot['data'],
             label='SIMI, Left Midfoot Vertical Velocity')

    # FVA Right Foot Peaks
    plt.scatter(FVA_Right_Foot['time'], FVA_Right_Foot['TO'],
                s=75, c='g')
    plt.scatter(FVA_Right_Foot['time'], FVA_Right_Foot['HS'],
                s=75, c='r')
    plt.plot(FVA_Right_Foot['time'], FVA_Right_Foot['data'],
             label='SIMI, Right Midfoot Vertical Velocity')

    # plt.plot(SIMIvars['time'], Heel_to_Heel[:, 1], label='Heel to Heel Distance')

    # SIMI instantaneous velocity
    plt.plot(SIMIvars['time'], SIMI_metrics['Spine_pos_deriv_velocity'],
             label='SIMI, Instantaneous Velocity (Spine Marker)')
    # GS First Contact Times, All Passes
    plt.scatter(GS_calc['First_Contact_time'], [1] * len(GS_calc['First_Contact_time']),
                label='GS, First Contact Time', s=75, c='gold')
    # GC Last Contact Times, All Passes
    plt.scatter(GS_calc['Last_Contact_time'], [1] * len(GS_calc['Last_Contact_time']),
                label='GS, Last Contact Time', s=75, c='black')

    plt.title('Foot Velocity Algorithm (FVA), GS and SIMI Metrics \n Participant (#%s) \n Trial GS: (%s) \n Trial SIMI: (%s)'
              % (Filenames['participant_num'], current_trial['GS_filename'][20:],
                 current_trial['SIMI_filename'][20:]), size=16)
    plt.xlabel('Time (s)', size=16)
    plt.ylabel('Velocity (m/s)', size=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend()
    lgd = ax.legend(handles, labels, loc='lower center', ncol=3,
                    bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()


_plot_FVA()

# %% Plot results of Heel-to-Heel


def _plot_HHD():
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    plt.show()
    ax.set_axisbelow(True)
    ax.yaxis.grid(color='0.95', linestyle='dashed')
    ax.xaxis.grid(color='0.95', linestyle='dashed')

    plt.plot(SIMIvars['time'], Heel_to_Heel[:, 1], label='Heel to Heel Distance')

    # SIMI instantaneous velocity
    plt.plot(SIMIvars['time'], SIMI_metrics['Spine_pos_deriv_velocity'],
             label='SIMI, Instantaneous Velocity (Spine Marker)')
    # GS First Contact Times, All Passes
    plt.scatter(GS_calc['First_Contact_time'], [1] * len(GS_calc['First_Contact_time']),
                label='GS, First Contact Time', s=75, c='gold')
    # GC Last Contact Times, All Passes
    plt.scatter(GS_calc['Last_Contact_time'], [1] * len(GS_calc['Last_Contact_time']),
                label='G,S Last Contact Time', s=75, c='black')

    plt.title('HHD Algorithm, GS and SIMI Velocities \n Participant (#%s) \n Trial SIMI: (%s)'
              % (Filenames['participant_num'],
                 current_trial['SIMI_filename'][20:]), size=20)
    plt.xlabel('Time (s)', size=20)
    plt.ylabel('Velocity (m/s)', size=20)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)

    handles, labels = ax.get_legend_handles_labels()
    plt.legend()
    lgd = ax.legend(handles, labels, loc='lower center', ncol=3,
                    bbox_to_anchor=(0.5, -0.35))
    plt.tight_layout()


_plot_HHD()


# %% Saving a Figure

# # Save the figure
# figname = ('GS_SIMI_PN%s_Trial_%s'
#            % (Filenames['participant_num'], current_trial['GS_filename'][20:-4]))
# figname = (Filenames['participant_dir']+'/'+figname)
# plt.savefig(figname)

# del figname
