# El desafío es poder procesar lo que emite Simian...
# main.py recogerá los datos en una sola mega DataFRAME

import json
from pathlib import Path
import pandas as pd
import glob2
import os

def read_Navi_Motion_File(archivo):
    # Abrir el archivo ".motion"... aqui lo modifique ya a .json
    with open(archivo, "r") as read_file:
        data = json.load(read_file)
    # ------------------------------------------------------------------------------------------------------
    # Ahora vamos des jsoneando el archivo para dejarlo en una base amigable para lo que es Pandas.
    # Esto está hecho ESPECIFICAMENTE para los archivos que emite
    df = pd.json_normalize(data, 'motionTrackDataPerTrials')
    df['Trial'] = range(len(df))
    # Mini Codigo para cambiar orden de columnas en un DataFrame de Pandas.
    order = [4, 0, 1, 2, 3]  # setting column's order
    df = df[[df.columns[i] for i in order]]
    # explode all columns with lists of dicts
    df = df.apply(lambda x: x.explode()).reset_index(drop=True)
    df.rename(columns={'positions': 'P'}, inplace=True)
    cols_to_normalize = ['P']

    # if there are keys, which will become column names, overlap with excising column names
    # add the current column name as a prefix
    normalized = list()
    for col in cols_to_normalize:
        d = pd.json_normalize(df[col], sep='_')
        d.columns = [f'{col}_{v}' for v in d.columns]
        normalized.append(d.copy())

    # combine df with the normalized columns
    df = pd.concat([df] + normalized, axis=1).drop(columns=cols_to_normalize)
    PosColumn = df.groupby(['Trial']).cumcount()
    PosCluymn = pd.Series(PosColumn)
    df = df.assign(Positions=PosCluymn)
    order = [0, 7, 4, 5, 6, 1, 2, 3]  # setting column's order
    df = df[[df.columns[i] for i in order]]

    # ------------------------------------------------------------------------------------------------------

    # Ahora transformaremos los datos de posición y tamaño de plataforma a "Pool Diameter"
    # como unidad de medida. Después de un analisis de un ensayo "vuelta olímpica, determinamos que
    # el Diametro de la piscina son 280x2 = 560 unidades virturales asi que:
    def normalizar_a_PD(uvirtuales):
        return (uvirtuales / 560)

    df[['P_position_x', 'P_position_y', 'platformPosition.x', 'platformPosition.y']] = df[['P_position_x', 'P_position_y', 'platformPosition.x', 'platformPosition.y']].apply(
        normalizar_a_PD)
    return df

def Asignar_TrueTrialsNames(row):
    Trial_Name = 'No asignado - error'
    #if row['']
    return Trial_Name

home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/" # Esto evidentemente varia. puede que no varié de compu a compu de Hayo
#----------------------------------------------------

Px_list = ['P06','P12'] # Lista de pacientes a incorporar en el análisis.

#----------------------------------------------------
Trial_uID = 0
df_full_list =[] # Lista que albergara el DataFrame completo de cada paciente
for Px in Px_list:
    Searchfiles = BaseDir + Px + '/SimianMaze_No_Inmersivo/*.motion' #Vamos a buscar los archivos Motion en el directorio de cada paciente
    Navi_files = glob2.glob(Searchfiles)
    Navi_files = sorted(Navi_files) #Esto ordena los archivos alfabeticamente
    df_list =[]   #preparamos una lista de los DataFrames que emergeran de cada archivo .motion
    for Navi_f in Navi_files:
        t_df = read_Navi_Motion_File(Navi_f)  #Aqui llamamos a nuesto super FUNCTION que definimos al principio del archivo y que des-json-iza un .motion a un Pandas DataFrame
        head, tail = os.path.split(Navi_f)  #Esto es para obtener solo el nombre del archivo y perder su directorio
        Bloque = tail[5] #esto es super especifico. el caracter [5] del nombre del archivo motion es A,B,C o D describiendo el bloque que usamos.
        Trial_uID +=100
        t_df.insert(1,'Origen',tail) #incorporamos el nombre del archivo de origen al DataFrame
        t_df.insert(1,'Trial Unique-ID',Trial_uID)
        t_df.insert(0,'Bloque',Bloque) #incorporamos el bloque de origen al DataFrame
        df_list.append(t_df) #añadimos el Dataframe a la lista que teniamo
        print('Anexado:  ',Px,'-',Bloque,'-',tail) #solo un reporte de como va la cosa.
    t_df = pd.concat(df_list) #juntamos todos los dataframes de un unico paciente.
    t_df.insert(0,'Sujeto',Px) #le añadimos una columna descriptora
    df_full_list.append(t_df) #lo añadimos a la megalista de los dataframes

df = pd.concat(df_full_list) #y juntamos todos los dataframes de todos los pacientes en un solo df




df.to_excel('MergedDataFrame.xlsx')  #y lo exportamos
print('Todo listo')