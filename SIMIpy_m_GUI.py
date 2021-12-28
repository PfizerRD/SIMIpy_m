from tkinter import *
import tkinter as tk

OPTIONS = [
    "Select Individual Trial",
    "Normal",
    "Fast",
    "Slow",
    "Carpet"
]

root = Tk()
root.geometry('400x400')
root.configure(background='#A2B5CD')
root.title('SIMI Trial Selector GUI')

trials_dropdown = StringVar(root)
trials_dropdown.set(OPTIONS[0])   # default value

filepairs = Variable(root)
trial = StringVar(root)

w = OptionMenu(root, trials_dropdown, *OPTIONS)
w.pack()
w.place(x=100, y=50)


def ok():

    print("Trial Selected: " + trials_dropdown.get())

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


def all_commands(): return [ok(), root.quit(), root.destroy()]


button = Button(root, text="OK", command=all_commands)
button.pack()
button.place(x=100, y=200)


mainloop()

temp = filepairs.get()
filepairs = temp
del temp

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

# Processing_filepairs .... list of all trials and associates filepairs
# filepairs .... set of three files in same trial (SIMI, GS, and GS Sync)
# filepath .... SIMI file path
# filepath_GS .... GS file path
# filepath_GS_sync .... GS PKMAS sync file path
