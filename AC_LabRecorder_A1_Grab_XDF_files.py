# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------


# El desafío es poder procesar lo que emite Lab Recorder

import json
from pathlib import Path
import pandas as pd
import glob2
import os
import pyxdf
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos

test_df = pd.read_csv('test_gaze.csv')


home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/"

FileName = BaseDir + 'P05/LSL_LAB/ses-NI/eeg/sub-P05_ses-NI_task-Default_run-001_eeg.xdf'

data, header = pyxdf.load_xdf(FileName)



t= data[0]['time_stamps']

def Extract(lst,place):
    return [item[place] for item in lst]

x= Extract(data[0]['time_series'],1)
LSL_df = pd.DataFrame(list(zip(t, x)), columns =['LSL_timestamp', 'LSL_norm_pos_x'])

test_df = test_df.loc[(test_df['gaze_timestamp']>3000) & (test_df['gaze_timestamp']<3100)]
ax = sns.lineplot(data= test_df, x= 'gaze_timestamp', y='norm_pos_x', alpha=1)


LSL_df = LSL_df.loc[(LSL_df['LSL_timestamp']>3000) & (LSL_df['LSL_timestamp']<3100)]
ax = sns.lineplot(data= LSL_df, x= 'LSL_timestamp', y='LSL_norm_pos_x', alpha=0.3)
plt.show()

print('hola')