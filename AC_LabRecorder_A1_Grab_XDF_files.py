# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------


# El desafío es poder procesar lo que emite Lab Recorder
#%%

import json
from pathlib import Path
import pandas as pd
import glob2
import os
import pyxdf
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/"

# Aqui incluimos un csv de pupil Labs de prueba...
test_df = pd.read_csv('test_gaze.csv')

# Aqui obtenemos un XDF de Lab Recorder de prueba
FileName = BaseDir + 'P05/LSL_LAB/ses-NI/eeg/sub-P05_ses-NI_task-Default_run-001_eeg.xdf'

# Extraemos los datos del XDF
data, header = pyxdf.load_xdf(FileName)
# data[0] es el Stream de Pupil Capture
# data[1] es el Stream de Overwatch


# Pensando en XDF-Labd Recorder Con esta función extraemos un elemento (en orden-place) de todo el Stream, un parametro... que están
# guardados en "time series" mientras que en "time stamps" esta la clave de sincronización de Lab Recorder
def Extract(lst,place):
    return [item[place] for item in lst]

#%%

t= data[0]['time_stamps']
x= Extract(data[0]['time_series'],1) # Stream pupilCapture, Canal 1: norm_pos_x
LSL_df = pd.DataFrame(list(zip(t, x)), columns =['LSL_timestamp', 'LSL_norm_pos_x'])
LSL_df = LSL_df.loc[(LSL_df['LSL_timestamp']>3000) & (LSL_df['LSL_timestamp']<3100)]
ax = sns.lineplot(data= LSL_df, x= 'LSL_timestamp', y='LSL_norm_pos_x', alpha=0.3)
plt.show()

#%%
time_stamp = data[1]['time_stamps']
MarkersAlfa = Extract(data[1]['time_series'],0) # Stream OverWatch Markes, Canal 0: Marker Primario
MarkersBeta = Extract(data[1]['time_series'],1) # Stream pupilCapture, Canal 1: Marker Secundario
Markers_df = pd.DataFrame(list(zip(time_stamp, MarkersAlfa, MarkersBeta)), columns =['OverWatch_time_stamp', 'OverWatch_MarkerA', 'OverWatch_MarkerB'])
MarkersA_df = Markers_df.loc[Markers_df['OverWatch_MarkerA']!='NONE']
#%%
# -------------------------------------------------------------
# Vamos a iniciar el análisis de los Markers de OverWatch para
# quedar con una lista confiable de marcadores

TimePoint = []
TimeStamp2 = []
Trial = []
LastTrial = 0
OnGoing = False
Ts = 0
print('-----------------------')
print('Inicio primera revisión')
print('-----------------------')

for row in MarkersA_df.itertuples():
    TP = 'NONE' # Para alimentar la lista de TimePoints
    Tr = 1000 # Para alimentar la lista de trials. Marcando un error
    Ts = row.OverWatch_time_stamp
    if row.OverWatch_MarkerA.isdigit():
        TP = 'START'
        Tr = int(row.OverWatch_MarkerA)
        LastTrial = Tr
        OnGoing = True
    if row.OverWatch_MarkerA == 'Stop':
        if OnGoing:
            TP = 'STOP'
            Tr = LastTrial
            OnGoing = False
    if row.OverWatch_MarkerA == 'Stop confirmado':
        if OnGoing: # Es caso de que no haya registro de un Stop previo
            TP = 'STOP'
            Tr = LastTrial
            OnGoing = False

    print(Ts,TP,Tr)
    TimeStamp2.append(Ts)
    TimePoint.append(TP)
    Trial.append(Tr)


#%%
test_df = test_df.loc[(test_df['gaze_timestamp']>3000) & (test_df['gaze_timestamp']<3100)]
ax = sns.lineplot(data= test_df, x= 'gaze_timestamp', y='norm_pos_x', alpha=1)




print('hola')