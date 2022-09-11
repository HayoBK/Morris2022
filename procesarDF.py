# ---------------------------------------------------------
# Lab ONCE - Septiembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------

import pandas as pd     #Base de datos
import numpy as np

m_df = pd.read_csv('MergedDataFrame.csv', index_col=0)
m_df.rename(columns = {'platformPosition.x':'platformPosition_x', 'platformPosition.y':'platformPosition_y'}, inplace = True)

for col in m_df.columns:
    print(col)
FirstRow=True
MegaList = []
for row in m_df.itertuples():
    if FirstRow: #En la primera linea de un Unique Trial
        AssesedTrial = row.Trial_Unique_ID
        FirstRow = False
        S=row.Sujeto
        M=row.Modalidad
        TB = row.True_Block
        TT = row.True_Trial
        ID = row.Trial_Unique_ID
        time = row.P_timeMilliseconds
        x = row.P_position_x
        y = row.P_position_y
        plat = row.platformExists
        platX = row.platformPosition_x
        playY = row.platformPosition_y
    if AssesedTrial != row.Trial_Unique_ID:  #Luego de que terminó el unique trial (hay que repetir esto al final del Loop.
        FirstRow = True
        rowList = [S,M,TB,TT,ID,time,plat] #x,y,plat,platX,playY]
        MegaList.append(rowList)
    # y aquí ponemos lo que ocurre en cada linea intermedia
    time=row.P_timeMilliseconds # voy updateando time para que quede "max" al terminar

rowList = [S,M,TB,TT,ID,time,plat] #x,y,plat,platX,playY]
MegaList.append(rowList)

short_df = pd.DataFrame(MegaList, columns = ['Sujeto','Modalidad','True_Block','True_Trial','Trial_Unique_ID','Duration(ms)','Platform_exists'])
print(short_df)
#Sujeto
#Modalidad
#True Block
#True Trial
#Bloque
#Trial
#Trial Unique-ID
#Origen
#Positions
#P_timeMilliseconds
#P_position_x
#P_position_y
#platformExists
#platformPosition.x
#platformPosition.y