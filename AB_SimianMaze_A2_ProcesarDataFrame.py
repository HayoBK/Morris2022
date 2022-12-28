# ---------------------------------------------------------
# Lab ONCE - Septiembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------

import pandas as pd     #Base de datos
import numpy as np
import matplotlib.pyplot as plt    #Graficos
import seaborn as sns   #Estetica de gráficos
pd.options.mode.chained_assignment = None  # default='warn'

m_df = pd.read_csv('AB_SimianMaze_Z1_RawMotion.csv', index_col=0)
m_df.rename(columns = {'platformPosition.x':'platformPosition_x', 'platformPosition.y':'platformPosition_y'}, inplace = True)
m_df = m_df[m_df.True_Trial < 8] #Aqui eliminamos esos primeros sujetos en que hicimios demasiados trials por bloques
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

#Hacer primer barrido para elminar los trials que hay que borrar
Banish_List=[]
for row in short_df.itertuples():
    if row.Path_length < 0.0005:
        Banish_List.append(row.Trial_Unique_ID)

#Aqui añadimos los Unique_IDd trials a borrar manualmente
Banish_List.extend([1201,2000,2100,1901])
Banish_List.extend([2700,3000,3100,3300,3400,3404,3005])
Banish_List.extend([3600,2901,2902,2903,4500,8500,7600,9300,11711,13102,13100,13600,14100,14300,14500,14600])
#a partir de P15 en adelante
Banish_List.extend([15500,15900,15604,15605,16600,22400,23606,24600,25400,25500,27604,27405,28500,])

# Aqui vamos a hacer una lista de los trials que eliminamos.
Banished_short_df = short_df[short_df['Trial_Unique_ID'].isin(Banish_List)]
Banished_short_df.to_excel('AB_SimianMaze_Z2_Banished_NaviData.xlsx')

Banished_long_df = m_df[m_df['Trial_Unique_ID'].isin(Banish_List)]
Banished_long_df.to_excel('AB_SimianMaze_Z2_Banished_NaviDataLong.xlsx')


short_df = short_df[~short_df['Trial_Unique_ID'].isin(Banish_List)]
m_df = m_df[~m_df['Trial_Unique_ID'].isin(Banish_List)]

# Aqui sacamos a sujetos especificos que se salieron de parametros o no cumplieron criterios de inclusion.
Banish_List =['P13']
short_df = short_df[~short_df['Sujeto'].isin(Banish_List)]
m_df = m_df[~m_df['Sujeto'].isin(Banish_List)]

safe_df = short_df # aqui estamos guardando la df para revisar cuales fueron los elementos eliminados por DropNa
short_df = short_df.dropna()
dropped_df = safe_df[~safe_df.index.isin(short_df.index)] # aqui vemos los elementos dropeados
dropped_df.to_excel('AB_SimianMaze_Z2_Dropped_NaviData.xlsx')

safe_df = m_df
m_df = m_df.dropna()
dropped_df = safe_df[~safe_df.index.isin(m_df.index)] # aqui vemos los elementos dropeados
dropped_df.to_excel('AB_SimianMaze_Z2_Dropped_NaviDataLong.xlsx')


#Limpieza completa.


#Iniciamos revisión manual de Trials repetidos por errores, para elegir que UniqueTrials añadir a la lista de Banish Manual.

e_df = short_df.groupby(['Sujeto','Modalidad','True_Block','True_Trial'])['Trial_Unique_ID'].apply(list).reset_index()
print(e_df)

print('Trials en conflicto!')
Conflict = False
for row in e_df.itertuples():
    if len(row.Trial_Unique_ID) > 1:
        Conflict = True
        print(row.Sujeto, row.Modalidad, row.True_Block, row.True_Trial, row.Trial_Unique_ID)
        show_df = m_df.loc[ (m_df['Sujeto']==row.Sujeto) & (m_df['Modalidad']==row.Modalidad) & (m_df['True_Block']==row.True_Block) & (m_df['True_Trial']==row.True_Trial)]
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="Trial_Unique_ID", data=show_df, linewidth=3, alpha=0.8, sort=False)  # palette = sns.color_palette('Blues', as_cmap = True),
        print('check')

        show_df = m_df.loc[(m_df['Sujeto'] == row.Sujeto) & (m_df['Modalidad'] == row.Modalidad) & (m_df['True_Block'] == row.True_Block) & (m_df['True_Trial'] == row.True_Trial)]
        show_df = show_df.reset_index()
        ax = sns.lineplot(x="P_position_x", y="P_position_y", hue="Trial_Unique_ID", data=show_df, linewidth=3, alpha=0.8, palette = sns.color_palette('Blues', as_cmap = True),sort=False)
        plt.show()
        show_df = short_df.loc[(short_df['Sujeto'] == row.Sujeto) & (short_df['Modalidad'] == row.Modalidad) & (short_df['True_Block'] == row.True_Block) & (short_df['True_Trial']
                                                                                                                                                             ==
                                                                                                                                               row.True_Trial)]
        ax = sns.barplot(x="Trial_Unique_ID",y="Path_length", data=show_df)
        plt.show()
        print('check')
print('¿Hubo conflicto? --> ',Conflict)

#-------------------------------------------------------------------------------------------------------------
#Ahora quiero corregir algunos errores puntuales interpolando datos especificos en Rows faltantes....
#-------------------------------------------------------------------------------------------------------------


def Interpolar_Row(Data,Suj,Bloq,TT): #Datos para localizar el Trial faltante
    Data = Data.reset_index(drop=True)
    Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']==Suj) & (Data['True_Block']==Bloq) & (Data['True_Trial']==TT-1)] # Obtenemos la linea inmediatament anterior
    Ind = Ind[0] # esto lo hacemos para transformar una lista en un número
    print('Intentando realizar interpolación de ',Suj,Bloq,TT)
    print('Obteniendo indice (debe ser un solo número)--> ',Ind)
    Copy_Row = Data.iloc[[Ind]]
    Promedio_CSE = (Data.iloc[Ind]['CSE'] + Data.iloc[Ind+1]['CSE'])/2
    Copy_Row.at[Ind,'CSE']=Promedio_CSE
    Copy_Row.at[Ind, 'True_Trial'] = TT
    print(Copy_Row)
    DataA = Data.iloc[:Ind, ]
    DataB = Data.iloc[Ind:, ]
    Data = DataA.append(Copy_Row).append(DataB).reset_index(drop=True)

    return Data

short_df = Interpolar_Row(short_df,'P12','HiddenTarget_3',2)
short_df = Interpolar_Row(short_df,'P07','VisibleTarget_2',3)
short_df = Interpolar_Row(short_df,'P19','HiddenTarget_3',3)
short_df = Interpolar_Row(short_df,'P21','HiddenTarget_3',2)
short_df = Interpolar_Row(short_df,'P28','HiddenTarget_3',2)

# Caso especial que no se puede usar Interpolar Row definido previamente:
#-------------------------------------------------------------------------------------------------------------
# Aqui se puede poner un primer TT

def Interpolar_Row_TT1_Missing(Data,Suj,Bloq,TT): #Datos para localizar el Trial faltante
    Data = Data.reset_index(drop=True)
    Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']==Suj) & (Data['True_Block']==Bloq) & (Data['True_Trial']==TT+1)] # Obtenemos la linea inmediatament siguiente en este caso
    Ind = Ind[0] # esto lo hacemos para transformar una lista en un número
    print('Intentando realizar interpolación de ',Suj,Bloq,TT)
    print('Obteniendo indice (debe ser un solo número)--> ',Ind)
    Copy_Row = Data.iloc[[Ind]]
    Promedio_CSE = Data.iloc[Ind]['CSE']
    Copy_Row.at[Ind,'CSE']=Promedio_CSE
    Copy_Row.at[Ind, 'True_Trial'] = TT
    print(Copy_Row)
    DataA = Data.iloc[:Ind, ]
    DataB = Data.iloc[Ind:, ]
    Data = DataA.append(Copy_Row).append(DataB).reset_index(drop=True)

    return Data

short_df = Interpolar_Row_TT1_Missing(short_df,'P21','VisibleTarget_2',1)
short_df = Interpolar_Row_TT1_Missing(short_df,'P11','HiddenTarget_3',1)

def Interpolar_Row_TT7_Missing(Data,Suj,Bloq,TT): #Datos para localizar el Trial faltante
    Data = Data.reset_index(drop=True)
    Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']==Suj) & (Data['True_Block']==Bloq) & (Data['True_Trial']==TT-1)] # Obtenemos la linea inmediatament anterior
    Ind = Ind[0] # esto lo hacemos para transformar una lista en un número
    print('Intentando realizar interpolación de ',Suj,Bloq,TT)
    print('Obteniendo indice (debe ser un solo número)--> ',Ind)
    Copy_Row = Data.iloc[[Ind]]
    Promedio_CSE = Data.iloc[Ind]['CSE']
    Copy_Row.at[Ind,'CSE']=Promedio_CSE
    Copy_Row.at[Ind, 'True_Trial'] = TT
    print(Copy_Row)
    DataA = Data.iloc[:Ind, ]
    DataB = Data.iloc[Ind:, ]
    Data = DataA.append(Copy_Row).append(DataB).reset_index(drop=True)

    return Data

short_df = Interpolar_Row_TT7_Missing(short_df,'P11','HiddenTarget_3',7)
short_df = Interpolar_Row_TT7_Missing(short_df,'P23','HiddenTarget_3',7)
short_df = Interpolar_Row_TT7_Missing(short_df,'P24','HiddenTarget_3',7)

short_df = Interpolar_Row_TT7_Missing(short_df,'P01','VisibleTarget_1',3)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','VisibleTarget_1',4)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',4)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',5)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',6)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_2',7)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_3',5)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_3',6)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','HiddenTarget_3',7)
short_df = Interpolar_Row_TT7_Missing(short_df,'P01','VisibleTarget_2',3)










#Repetir V2 desde V1 en P05
Data= short_df
Data = Data.reset_index(drop=True)
Copy_Row = Data.loc[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']=='P05') & (Data['True_Block']=='VisibleTarget_1') & (Data['True_Trial']<4)]
Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']=='P05') & (Data['True_Block']=='VisibleTarget_1') & (Data['True_Trial']==1)]
Ind = Ind[0]
for i in range(3):
    Copy_Row.at[(Ind+i), 'True_Block'] = 'VisibleTarget_2'
Ind = Data.index[(Data['Modalidad']=='No Inmersivo') & (Data['Sujeto']=='P05') & (Data['True_Block']=='HiddenTarget_3') & (Data['True_Trial']==7)]
Ind = Ind[0]+1
DataA = short_df.iloc[:Ind, ]
DataB = short_df.iloc[Ind:, ]
short_df = DataA.append(Copy_Row).append(DataB).reset_index(drop=True)


#-------------------------------------------------------------------------------------------------------------

# Ahora enriquecemos la base de datos con datos de los pacientes

#-------------------------------------------------------------------------------------------------------------

codex = pd.read_excel('AA_CODEX.xlsx',index_col=0)
Codex_Dict = codex.to_dict('series')
short_df['Edad'] = short_df['Sujeto']
short_df['Edad'].replace(Codex_Dict['Edad'], inplace=True)
m_df['Edad'] = m_df['Sujeto']
m_df['Edad'].replace(Codex_Dict['Edad'], inplace=True)
short_df['Genero'] = short_df['Sujeto']
short_df['Genero'].replace(Codex_Dict['Genero'], inplace=True)
m_df['Genero'] = m_df['Sujeto']
m_df['Genero'].replace(Codex_Dict['Genero'], inplace=True)
short_df['Dg'] = short_df['Sujeto']
short_df['Dg'].replace(Codex_Dict['Dg'], inplace=True)
m_df['Dg'] = m_df['Sujeto']
m_df['Dg'].replace(Codex_Dict['Dg'], inplace=True)
short_df['Grupo'] = short_df['Sujeto']
short_df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)
move = short_df.pop('Grupo')
short_df.insert(1,'Grupo',move)
m_df['Grupo'] = m_df['Sujeto']
m_df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)
move = m_df.pop('Grupo')
m_df.insert(1,'Grupo',move)
print(short_df)

#Aqui hare una DataBase Limpia de los sujetos incluidos en el estudio, solo datos básicos
PXX_List = short_df['Sujeto'].unique()  # Obtenemos una lista de todos los sujetos individualizados
codex_df = pd.DataFrame(PXX_List, columns = ['Sujeto'])
codex_df['Edad'] = codex_df['Sujeto']
codex_df['Edad'].replace(Codex_Dict['Edad'], inplace=True)
codex_df['Grupo'] = codex_df['Sujeto']
codex_df['Grupo'].replace(Codex_Dict['Grupo'], inplace=True)

Resumen_codex_df = codex_df.groupby('Grupo')['Edad'].agg(Conteo='size', Edad_promedio='mean').reset_index()


m_df.to_csv('AB_SimianMaze_Z2_NaviData_con_posicion.csv')
print('25%')
#m_df.to_excel('AB_SimianMaze_Z2_NaviData_con_posicion.xlsx')
print('50%')
short_df.to_csv('AB_SimianMaze_Z3_NaviDataBreve_con_calculos.csv')
print('75%')
#short_df.to_excel('AB_SimianMaze_Z3_NaviDataBreve_con_calculos.xlsx')
print('100% todo listo ')