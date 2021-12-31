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
        plt.plot(mean, p(mean), "-", color=mycolor)
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


# %% Bland Altman Plots for SIMI HS (FVA) vs GS HS

def _FVA_HS_Bland_Altman(Participants_HS_TO,
                         HeelStrike_SIMI, HeelStrike_GS):
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


_FVA_HS_Bland_Altman(Participants_HS_TO, HeelStrike_SIMI, HeelStrike_GS)

# %% Bland Altman Plots for SIMI TO (FVA) vs GS TO


def _FVA_TO_Bland_Altman(Participants_HS_TO, ToeOff_SIMI, ToeOff_GS):
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


_FVA_TO_Bland_Altman(Participants_HS_TO, ToeOff_SIMI, ToeOff_GS)


# %% Bland Altman Plots - Compile all Patricipant Data, then plot using:
# Polt definitions:    _FVA_HS_Bland_Altman():
#                      _FVA_TO_Bland_Altman():

All_HS_TO = {"HeelStrike_SIMI_Normal": [],
             "HeelStrike_SIMI_Fast": [],
             "HeelStrike_SIMI_Slow": [],
             "HeelStrike_SIMI_Carpet": [],
             "ALL_HS_SIMI": [],

             "ToeOff_SIMI_Normal": [],
             "ToeOff_SIMI_Fast": [],
             "ToeOff_SIMI_Slow": [],
             "ToeOff_SIMI_Carpet": [],
             "ALL_TO_SIMI": [],

             "HeelStrike_GS_Normal": [],
             "HeelStrike_GS_Fast": [],
             "HeelStrike_GS_Slow": [],
             "HeelStrike_GS_Carpet": [],
             "ALL_HS_GS": [],

             "ToeOff_GS_Normal": [],
             "ToeOff_GS_Fast": [],
             "ToeOff_GS_Slow": [],
             "ToeOff_GS_Carpet": [],
             "ALL_TO_GS": []
             }

for n in Participants_HS_TO:
    print(n)
    All_HS_TO["HeelStrike_SIMI_Normal"].append(Participants_HS_TO[n]["HeelStrike_SIMI_Normal"])
    All_HS_TO["HeelStrike_SIMI_Fast"].append(Participants_HS_TO[n]["HeelStrike_SIMI_Fast"])
    All_HS_TO["HeelStrike_SIMI_Slow"].append(Participants_HS_TO[n]["HeelStrike_SIMI_Slow"])
    All_HS_TO["HeelStrike_SIMI_Carpet"].append(Participants_HS_TO[n]["HeelStrike_SIMI_Carpet"])

    All_HS_TO["ToeOff_SIMI_Normal"].append(Participants_HS_TO[n]["ToeOff_SIMI_Normal"])
    All_HS_TO["ToeOff_SIMI_Fast"].append(Participants_HS_TO[n]["ToeOff_SIMI_Fast"])
    All_HS_TO["ToeOff_SIMI_Slow"].append(Participants_HS_TO[n]["ToeOff_SIMI_Slow"])
    All_HS_TO["ToeOff_SIMI_Carpet"].append(Participants_HS_TO[n]["ToeOff_SIMI_Carpet"])

    All_HS_TO["HeelStrike_GS_Normal"].append(Participants_HS_TO[n]["HeelStrike_GS_Normal"])
    All_HS_TO["HeelStrike_GS_Fast"].append(Participants_HS_TO[n]["HeelStrike_GS_Fast"])
    All_HS_TO["HeelStrike_GS_Slow"].append(Participants_HS_TO[n]["HeelStrike_GS_Slow"])
    All_HS_TO["HeelStrike_GS_Carpet"].append(Participants_HS_TO[n]["HeelStrike_GS_Carpet"])

    All_HS_TO["ToeOff_GS_Normal"].append(Participants_HS_TO[n]["ToeOff_GS_Normal"])
    All_HS_TO["ToeOff_GS_Fast"].append(Participants_HS_TO[n]["ToeOff_GS_Fast"])
    All_HS_TO["ToeOff_GS_Slow"].append(Participants_HS_TO[n]["ToeOff_GS_Slow"])
    All_HS_TO["ToeOff_GS_Carpet"].append(Participants_HS_TO[n]["ToeOff_GS_Carpet"])


All_HS_TO["HeelStrike_SIMI_Normal"] = np.concatenate(All_HS_TO["HeelStrike_SIMI_Normal"])
All_HS_TO["HeelStrike_SIMI_Fast"] = np.concatenate(All_HS_TO["HeelStrike_SIMI_Fast"])
All_HS_TO["HeelStrike_SIMI_Slow"] = np.concatenate(All_HS_TO["HeelStrike_SIMI_Slow"])
All_HS_TO["HeelStrike_SIMI_Carpet"] = np.concatenate(All_HS_TO["HeelStrike_SIMI_Carpet"])
All_HS_TO["ALL_HS_SIMI"] = np.concatenate([All_HS_TO["HeelStrike_SIMI_Normal"],
                                          All_HS_TO["HeelStrike_SIMI_Fast"],
                                          All_HS_TO["HeelStrike_SIMI_Slow"],
                                          All_HS_TO["HeelStrike_SIMI_Carpet"]])

All_HS_TO["ToeOff_SIMI_Normal"] = np.concatenate(All_HS_TO["ToeOff_SIMI_Normal"])
All_HS_TO["ToeOff_SIMI_Fast"] = np.concatenate(All_HS_TO["ToeOff_SIMI_Fast"])
All_HS_TO["ToeOff_SIMI_Slow"] = np.concatenate(All_HS_TO["ToeOff_SIMI_Slow"])
All_HS_TO["ToeOff_SIMI_Carpet"] = np.concatenate(All_HS_TO["ToeOff_SIMI_Carpet"])
All_HS_TO["ALL_TO_SIMI"] = np.concatenate([All_HS_TO["ToeOff_SIMI_Normal"],
                                           All_HS_TO["ToeOff_SIMI_Fast"],
                                           All_HS_TO["ToeOff_SIMI_Slow"],
                                           All_HS_TO["ToeOff_SIMI_Carpet"]])

All_HS_TO["HeelStrike_GS_Normal"] = np.concatenate(All_HS_TO["HeelStrike_GS_Normal"])
All_HS_TO["HeelStrike_GS_Fast"] = np.concatenate(All_HS_TO["HeelStrike_GS_Fast"])
All_HS_TO["HeelStrike_GS_Slow"] = np.concatenate(All_HS_TO["HeelStrike_GS_Slow"])
All_HS_TO["HeelStrike_GS_Carpet"] = np.concatenate(All_HS_TO["HeelStrike_GS_Carpet"])
All_HS_TO["ALL_HS_GS"] = np.concatenate([All_HS_TO["HeelStrike_GS_Normal"],
                                        All_HS_TO["HeelStrike_GS_Fast"],
                                        All_HS_TO["HeelStrike_GS_Slow"],
                                        All_HS_TO["HeelStrike_GS_Carpet"]])

All_HS_TO["ToeOff_GS_Normal"] = np.concatenate(All_HS_TO["ToeOff_GS_Normal"])
All_HS_TO["ToeOff_GS_Fast"] = np.concatenate(All_HS_TO["ToeOff_GS_Fast"])
All_HS_TO["ToeOff_GS_Slow"] = np.concatenate(All_HS_TO["ToeOff_GS_Slow"])
All_HS_TO["ToeOff_GS_Carpet"] = np.concatenate(All_HS_TO["ToeOff_GS_Carpet"])
All_HS_TO["ALL_TO_GS"] = np.concatenate([All_HS_TO["ToeOff_GS_Normal"],
                                        All_HS_TO["ToeOff_GS_Fast"],
                                        All_HS_TO["ToeOff_GS_Slow"],
                                        All_HS_TO["ToeOff_GS_Carpet"]])


# %% Plot ALL PARTICIPANT FVA results


def _FVA_ALL_Bland_Altman():
    plt.figure()
    # position the figure:
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(300, 50, 1100, 1000)

    # HS ~~~~~~~~~~~~~~~~~~~~~~
    # All Trials
    bland_altman_plot(All_HS_TO["ALL_HS_GS"][:, 0], All_HS_TO["ALL_HS_SIMI"][:, 0], 'gray', 'All')

    # Normal Trial
    bland_altman_plot(All_HS_TO["HeelStrike_GS_Normal"][:, 0], All_HS_TO["HeelStrike_SIMI_Normal"][:, 0], 'green', 'Normal Pace')

    # Fast Trial
    bland_altman_plot(All_HS_TO["HeelStrike_GS_Fast"][:, 0], All_HS_TO["HeelStrike_SIMI_Fast"][:, 0], 'magenta', 'Fast Pace')

    # # Slow Trial
    bland_altman_plot(All_HS_TO["HeelStrike_GS_Slow"][:, 0], All_HS_TO["HeelStrike_SIMI_Slow"][:, 0], 'blue', 'Slow Pace')

    # # Carpeted Trial
    bland_altman_plot(All_HS_TO["HeelStrike_GS_Carpet"][:, 0], All_HS_TO["HeelStrike_SIMI_Carpet"][:, 0], 'black', 'Carpeted')

    plt.title('Bland-Altman Plot: GS HS vs SIMI HS (FVA) \n ALL Participants', size=16)
    plt.xlabel('Mean HS Time (GS + SIMI) (s)', size=16)
    plt.ylabel('Difference in HS Time (GS - SIMI) (s)', size=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.ylim([-0.15, 0.15])
    plt.legend(bbox_to_anchor=(0, -0.15, 1, 1), loc='lower center', ncol=5, prop={'size': 16})
    plt.tight_layout()
    plt.show()

    plt.figure()
    # position the figure:
    mngr = plt.get_current_fig_manager()
    mngr.window.setGeometry(300, 50, 1100, 1000)

    # TO ~~~~~~~~~~~~~~~~~~~~~~
    # All Trials
    bland_altman_plot(All_HS_TO["ALL_TO_GS"][:, 0], All_HS_TO["ALL_TO_SIMI"][:, 0], 'gray', 'All')

    # Normal Trial
    bland_altman_plot(All_HS_TO["ToeOff_GS_Normal"][:, 0], All_HS_TO["ToeOff_SIMI_Normal"][:, 0], 'green', 'Normal Pace')

    # Fast Trial
    bland_altman_plot(All_HS_TO["ToeOff_GS_Fast"][:, 0], All_HS_TO["ToeOff_SIMI_Fast"][:, 0], 'magenta', 'Fast Pace')

    # # Slow Trial
    bland_altman_plot(All_HS_TO["ToeOff_GS_Slow"][:, 0], All_HS_TO["ToeOff_SIMI_Slow"][:, 0], 'blue', 'Slow Pace')

    # # Carpeted Trial
    bland_altman_plot(All_HS_TO["ToeOff_GS_Carpet"][:, 0], All_HS_TO["ToeOff_SIMI_Carpet"][:, 0], 'black', 'Carpeted')

    plt.title('Bland-Altman Plot: GS TO vs SIMI TO (FVA) \n ALL Participants', size=16)
    plt.xlabel('Mean TO Time (GS + SIMI) (s)', size=16)
    plt.ylabel('Difference in TO Time (GS - SIMI) (s)', size=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.ylim([-0.15, 0.15])
    plt.legend(bbox_to_anchor=(0, -0.15, 1, 1), loc='lower center', ncol=5, prop={'size': 16})
    plt.tight_layout()
    plt.show()


_FVA_ALL_Bland_Altman()


# %% Scatter Plot (GS in x-axis vs SIMI in y-axis)

# def scatter_plot(dataGS, dataSIMI, mycolor, trial_type):

#     if trial_type == 'All':
#         dataGS = np.asarray(dataGS, dtype='float64')
#         dataSIMI = np.asarray(dataSIMI, dtype='float64')

#         z = np.polyfit(dataGS, dataSIMI, 1)
#         p = np.poly1d(z)
#         y_hat = np.poly1d(z)(dataGS)
#         plt.plot(dataGS, y_hat, "-", lw=1, color=mycolor, label=trial_type)
#         text = f"$y={z[0]:0.3f}\;dataGS{z[1]:+0.3f}$\n$R^2 = {r2_score(dataSIMI,y_hat):0.3f}$"
#         plt.gca().text(0.05, 0.95, text, transform=plt.gca().transAxes,
#                         fontsize=14, verticalalignment='top')
#         # Plot ideal 100% match line for GS data (i.e. GS Data vs. GS Data)
#         plt.plot(dataGS, dataGS, '-', color='blue', label='Unit Line (GS=SIMI)')

#     if trial_type != "All":
#         dataGS = np.asarray(dataGS, dtype='float64')
#         dataSIMI = np.asarray(dataSIMI, dtype='float64')
#         plt.scatter(dataGS, dataSIMI, color=mycolor, s=75, label=trial_type)


# plt.figure()

# # All Trials
# scatter_plot(HeelStrike_GS['All'][:, 0], HeelStrike_SIMI['All'][:, 0], 'gray', 'All')

# # Normal Trial
# scatter_plot(HeelStrike_GS['Normal'][:, 0], HeelStrike_SIMI['Normal'][:, 0], 'green', 'Normal Pace')

# # Fast Trial
# scatter_plot(HeelStrike_GS['Fast'][:, 0], HeelStrike_SIMI['Fast'][:, 0], 'magenta', 'Fast Pace')

# # Slow Trial
# scatter_plot(HeelStrike_GS['Slow'][:, 0], HeelStrike_SIMI['Slow'][:, 0], 'blue', 'Slow Pace')

# # Carpeted Trial
# scatter_plot(HeelStrike_GS['Carpet'][:, 0], HeelStrike_SIMI['Carpet'][:, 0], 'black', 'Carpeted')


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
