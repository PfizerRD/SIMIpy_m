from tkinter import *
import tkinter as tk
from SIMIpy_m_main import _Participant_Metrics
from SIMIpy_m_metrics import _metrics
from SIMIpy_m_filenames import _filenames
from SIMIpy_m_processing_filepair import _filepair_chooser
from pathlib import Path

OPTIONS = [
    "Select Individual Trial",
    "Normal",
    "Fast",
    "Slow",
    "Carpet"
]

OPTIONS_PN = [
    "Select Participant",
    "10010002",
    "10010003",
    "10010004",
    "10010005",
    "10010006",
    "10010007",
    "10010008",
    "10010009",
    "10010010",
    "10010011",
    "10010012",
    "10010013",
    "10010014",
    "10010015",
    "10010016",
    "10010017",
    "10010018",
    "10010019",
    "10010020"
]

root = Tk()
root.geometry('275x200')
root.configure(background='#A2B5CD')
root.title('SIMI Trial Selector GUI')

trials_dropdown = StringVar(root)
trials_dropdown.set(OPTIONS[0])   # default value

participants_dropdown = StringVar(root)
participants_dropdown.set(OPTIONS_PN[0])   # default value

filepairs = Variable(root)
trial = StringVar(root)

w = OptionMenu(root, trials_dropdown, *OPTIONS)
w.pack()
w.place(x=25, y=75)

w2 = OptionMenu(root, participants_dropdown, *OPTIONS_PN)
w2.pack()
w2.place(x=25, y=25)


def ok():

    print("Participant Selected: " + participants_dropdown.get())
    print("Trial Selected: " + trials_dropdown.get())


def all_commands(): return [ok(), root.quit(), root.destroy()]


button = Button(root, text="OK", command=all_commands)
button.pack()
button.place(x=25, y=150)


mainloop()

pn = participants_dropdown.get()
Current_Participant_Path = str(Path(parentpath, pn))
Filenames = {}
Filenames = _filenames(Filenames, Current_Participant_Path)
Processing_filepairs = _filepair_chooser(Filenames)


if trials_dropdown.get() == 'Normal':
    filepairs.set(Processing_filepairs[0])
    trial.set('Normal')

elif trials_dropdown.get() == 'Fast':
    filepairs.set(Processing_filepairs[1])
    trial.set('Fast')

elif trials_dropdown.get() == 'Slow':
    filepairs.set(Processing_filepairs[2])
    trial.set('Slow')

elif trials_dropdown.get() == 'Carpet':
    filepairs.set(Processing_filepairs[3])
    trial.set('Carpet')

filepairs = filepairs.get()

for n in range(0, len(filepairs)):
    index_sync = filepairs[n].find('PKMAS_Sync')
    index_SIMI = filepairs[n].find('SIMI')

    if index_SIMI != -1:
        filepath = filepairs[n]
    elif index_sync != -1:
        filepath_GS_sync = filepairs[n]
    else:
        filepath_GS = filepairs[n]
    del index_sync, index_SIMI

[Participants_HS_TO, Batch_Outputs,
 HeelStrike_SIMI, HeelStrike_GS,
 ToeOff_SIMI, ToeOff_GS, SIMI_metrics,
 GS_calc, GS_vars, GS_PKMAS_sync, FVA_vars,
 FVA_Left_Foot, FVA_Right_Foot,
 SIMIvars, current_trial] = _Participant_Metrics(Filenames, Processing_filepairs)


[SIMI_metrics, GS_calc, GS_vars,
 GS_PKMAS_sync, FVA_vars,
 FVA_Left_Foot, FVA_Right_Foot, HHD_calc,
 SIMIvars, SIMIvars_original,
 current_trial] = _metrics(filepath, filepath_GS,
                           filepath_GS_sync, trial)


# Processing_filepairs .... list of all trials and associates filepairs
# filepairs .... set of three files in same trial (SIMI, GS, and GS Sync)
# filepath .... SIMI file path
# filepath_GS .... GS file path
# filepath_GS_sync .... GS PKMAS sync file path
