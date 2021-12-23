# %% Bland-Altman Plot

import matplotlib.pyplot as plt
import numpy as np
# import pandas
from sklearn.metrics import r2_score


def bland_altman_plot(dataGS, dataSIMI, mycolor, trial_type):

    # dataGS = np.array(dataGS[:, 0], dtype='float64')
    # dataSIMI = np.array(dataSIMI[:, 0], dtype='float64')

    if trial_type == 'All':
        dataGS = np.asarray(dataGS, dtype='float64')
        dataSIMI = np.asarray(dataSIMI, dtype='float64')
        mean = np.nanmean([dataGS, dataSIMI], axis=0)
        diff = dataGS - dataSIMI          # Difference between dataGS and dataSIMI
        md = np.nanmean(diff)             # Mean of the difference
        sd = np.std(diff, axis=0)         # Standard deviation of the difference

        # Plot mean line, and (+/-) 2 * S.D. lines
        plt.axhline(md,           color='gray', linestyle='--')
        plt.axhline(md + 2*sd, color='gray', linestyle='--')
        plt.axhline(md - 2*sd, color='gray', linestyle='--')

        z = np.polyfit(mean, diff, 1)
        p = np.poly1d(z)
        plt.plot(mean, p(mean), "--", color=mycolor)
        y_hat = np.poly1d(z)(mean)
        plt.plot(mean, y_hat, "--", lw=1, color=mycolor, label=trial_type)
        text = f"$y={z[0]:0.3f}\;mean{z[1]:+0.3f}$\n$R^2 = {r2_score(diff,y_hat):0.3f}$"
        plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
                       fontsize=14, verticalalignment='top')

    if trial_type != "All":
        dataGS = np.asarray(dataGS, dtype='float64')
        dataSIMI = np.asarray(dataSIMI, dtype='float64')
        mean = np.nanmean([dataGS, dataSIMI], axis=0)
        diff = dataGS - dataSIMI          # Difference between dataGS and dataSIMI
        md = np.nanmean(diff)             # Mean of the difference
        sd = np.std(diff, axis=0)         # Standard deviation of the difference

        plt.scatter(mean, diff, color=mycolor, s=75, label=trial_type)

# The corresponding elements in dataGS and dataSIMI are used to calculate the coordinates
# for the plotted points. Then, create a plot by running the code below.


# %% Bland Altman Plots for HS


plt.figure()

# position the figure:
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(300, 50, 1100, 1000)

# All Trials
bland_altman_plot(HeelStrike_GS['All'][:, 0], HeelStrike_SIMI['All'][:, 0], 'gray', 'All')

# Normal Trial
bland_altman_plot(HeelStrike_GS['Normal'][:, 0], HeelStrike_SIMI['Normal'][:, 0], 'green', 'Normal Pace')

# Fast Trial
bland_altman_plot(HeelStrike_GS['Fast'][:, 0], HeelStrike_SIMI['Fast'][:, 0], 'magenta', 'Fast Pace')

# Slow Trial
bland_altman_plot(HeelStrike_GS['Slow'][:, 0], HeelStrike_SIMI['Slow'][:, 0], 'blue', 'Slow Pace')

# Carpeted Trial
bland_altman_plot(HeelStrike_GS['Carpet'][:, 0], HeelStrike_SIMI['Carpet'][:, 0], 'black', 'Carpeted')

plt.title('Bland-Altman Plot: GS HS vs SIMI HS (FVA) \n Participant (#%s)'
          % (Filenames['participant_num']), size=16)
plt.xlabel('Mean HS Time (GS + SIMI) (s)', size=16)
plt.ylabel('Difference in HS Time (GS - SIMI) (s)', size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim([-0.15, 0.15])
plt.legend(bbox_to_anchor=(0, -0.15, 1, 1), loc='lower center', ncol=5, prop={'size': 16})
plt.tight_layout()
plt.show()


# %% Bland Altman Plots for TO

plt.figure()

# position the figure:
mngr = plt.get_current_fig_manager()
mngr.window.setGeometry(300, 50, 1100, 1000)

# All Trials
bland_altman_plot(ToeOff_GS['All'][:, 0], ToeOff_SIMI['All'][:, 0], 'gray', 'All')

# Normal Trial
bland_altman_plot(ToeOff_GS['Normal'][:, 0], ToeOff_SIMI['Normal'][:, 0], 'green', 'Normal Pace')

# Fast Trial
bland_altman_plot(ToeOff_GS['Fast'][:, 0], ToeOff_SIMI['Fast'][:, 0], 'magenta', 'Fast Pace')

# Slow Trial
bland_altman_plot(ToeOff_GS['Slow'][:, 0], ToeOff_SIMI['Slow'][:, 0], 'blue', 'Slow Pace')

# Carpeted Trial
bland_altman_plot(ToeOff_GS['Carpet'][:, 0], ToeOff_SIMI['Carpet'][:, 0], 'black', 'Carpeted')


plt.title('Bland-Altman Plot: GS TO vs SIMI TO (FVA) \n Participant (#%s)'
          % (Filenames['participant_num']), size=16)
plt.xlabel('Mean TO Time (GS + SIMI) (s)', size=16)
plt.ylabel('Difference in TO Time (GS - SIMI) (s)', size=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
# plt.ylim([-0.15, 0.15])
plt.legend(bbox_to_anchor=(0, -0.15, 1, 1), loc='lower center', ncol=5, prop={'size': 16})
plt.tight_layout()
plt.show()

# %% Scatter Plot (GS in x-axis vs SIMI in y-axis)

# def scatter_plot(dataGS, dataSIMI, mycolor, trial_type):

#     if trial_type == 'All':
#         dataGS = np.asarray(dataGS)
#         dataSIMI = np.asarray(dataSIMI)

#         z = np.polyfit(dataGS, dataSIMI, 1)
#         p = np.poly1d(z)
#         y_hat = np.poly1d(z)(dataGS)
#         plt.plot(dataGS, y_hat, "-", lw=1, color=mycolor, label=trial_type)
#         text = f"$y={z[0]:0.3f}\;dataGS{z[1]:+0.3f}$\n$R^2 = {r2_score(dataSIMI,y_hat):0.3f}$"
#         plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
#                        fontsize=14, verticalalignment='top')
#         # Plot ideal 100% match line for GS data (i.e. GS Data vs. GS Data)
#         plt.plot(dataGS, dataGS, '-', color='blue', label='Unit Line (GS=SIMI)')

#     if trial_type != "All":
#         dataGS = np.asarray(dataGS)
#         dataSIMI = np.asarray(dataSIMI)
#         plt.scatter(dataGS, dataSIMI, color=mycolor, s=75, label=trial_type)


# plt.figure()

# # All Trials
# scatter_plot(HeelStrike_GS['All'], HeelStrike_SIMI['All'], 'gray', 'All')

# # Normal Trial
# scatter_plot(HeelStrike_GS['Normal'], HeelStrike_SIMI['Normal'], 'green', 'Normal Pace')

# # Fast Trial
# scatter_plot(HeelStrike_GS['Fast'], HeelStrike_SIMI['Fast'], 'magenta', 'Fast Pace')

# # Slow Trial
# scatter_plot(HeelStrike_GS['Slow'], HeelStrike_SIMI['Slow'], 'blue', 'Slow Pace')

# # Carpeted Trial
# scatter_plot(HeelStrike_GS['Carpet'], HeelStrike_SIMI['Carpet'], 'black', 'Carpeted')


# # position the figure:
# mngr = plt.get_current_fig_manager()
# mngr.window.setGeometry(300, 50, 1000, 1000)

# plt.title('GS HS vs SIMI HS (FVA)', fontsize=20)
# plt.xlabel('GS HS Time (s)', fontsize=20)
# plt.ylabel('SIMI HS Time (s)', fontsize=20)
# plt.xticks(fontsize=16)
# plt.yticks(fontsize=16)
# plt.legend(bbox_to_anchor=(0, -0.225, 1, 1), loc='lower center', ncol=3, prop={'size': 16})
# plt.axis('square')
# plt.tight_layout()
# plt.show()
