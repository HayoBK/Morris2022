# ---------------------------------------------------------
# Lab ONCE - Septiembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------

import pandas as pd     #Base de datos
import numpy as np
import matplotlib.pyplot as plt    #Graficos
import seaborn as sns   #Estetica de gráficos

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
        path_length = 0
        plat = row.platformExists
        platX = row.platformPosition_x
        platY = row.platformPosition_y

        #Cálculo de CSE
        Time_Track = 100
        x_Track = x
        y_Track = y
        Measures_Track = 1
        CSE = 0

    if AssesedTrial != row.Trial_Unique_ID:  #Luego de que terminó el unique trial (hay que repetir esto al final del Loop.
        FirstRow = True
        rowList = [S,M,TB,TT,ID,time,path_length,CSE,plat] #x,y,plat,platX,playY]
        MegaList.append(rowList)

    # y aquí ponemos lo que ocurre en cada linea intermedia
    time=row.P_timeMilliseconds # voy updateando time para que quede "max" al terminar
    path_length+=((x - row.P_position_x)**2 + (y - row.P_position_y)**2)**0.5
    x = row.P_position_x
    y = row.P_position_y

    # Cálculo de CSE
    x_Track += x
    y_Track += y
    Measures_Track += 1

    if time > Time_Track:
        Time_Track += 100
        AvgX = x_Track / Measures_Track
        AvgY = y_Track / Measures_Track
        CSE += ((AvgX - platX)**2 + (AvgY - platY)**2)**0.5


#Aqui tenemos que repetir el "rowList" para dar cuenta del ultimo trial del loop
rowList = [S,M,TB,TT,ID,time,path_length,CSE,plat] #x,y,plat,platX,playY]
MegaList.append(rowList)

short_df = pd.DataFrame(MegaList, columns = ['Sujeto','Modalidad','True_Block','True_Trial','Trial_Unique_ID','Duration(ms)','Path_length','CSE','Platform_exists'])
print(short_df)

#Hacer primer barrido para elminar los trials que hay que borrar
Banish_List=[]
for row in short_df.itertuples():
    if row.Path_length < 0.0005:
        Banish_List.append(row.Trial_Unique_ID)
print(Banish_List)

Banish_List.extend([1201])

short_df = short_df[~short_df['Trial_Unique_ID'].isin(Banish_List)]

e_df = short_df.groupby(['Sujeto','Modalidad','True_Block','True_Trial'])['Trial_Unique_ID'].apply(list).reset_index()
print(e_df)

print('Trials en conflicto!')
for row in e_df.itertuples():
    if len(row.Trial_Unique_ID) > 1:
        print(row.Sujeto, row.Modalidad, row.True_Block, row.True_Trial, row.Trial_Unique_ID)
        show_df = m_df.loc[ (m_df['Sujeto']==row.Sujeto) & (m_df['Modalidad']==row.Modalidad) & (m_df['True_Block']==row.True_Block) & (m_df['True_Trial']==row.True_Trial)]
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="Trial_Unique_ID", data=show_df, linewidth=3, alpha=0.8, sort=False)  # palette = sns.color_palette('Blues', as_cmap = True),
        print('check')
Subj='P06'
Mod ='Realidad Virtual'
Plat=True
show_df = short_df.loc[ (short_df['Sujeto']==Subj) & (short_df['Modalidad']==Mod)]
ax = sns.barplot(x="True_Block", y="Path_length",hue="True_Trial",data=show_df)
plt.show()
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