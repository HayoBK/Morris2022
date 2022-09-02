# El desafío es poder procesar lo que emite Simian...
# main.py recogerá los datos en una sola mega DataFRAME

import json

import pandas as pd

#Abrir el archivo ".motion"... aqui lo modifique ya a .json
with open("Navi360.motion", "r") as read_file:
    data = json.load(read_file)

#------------------------------------------------------------------------------------------------------
#Ahora vamos des jsoneando el archivo para dejarlo en una base amigable para lo que es Pandas.
#Esto está hecho ESPECIFICAMENTE para los archivos que emite
df = pd.json_normalize(data,'motionTrackDataPerTrials')
df['Trial'] = range(len(df))
#Mini Codigo para cambiar orden de columnas en un DataFrame de Pandas.
order = [4,0,1,2,3] # setting column's order
df = df[[df.columns[i] for i in order]]
# explode all columns with lists of dicts
df = df.apply(lambda x: x.explode()).reset_index(drop=True)
df.rename(columns={'positions':'P'}, inplace=True)
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
df = df.assign(Positions = PosCluymn)
order = [0,7,4,5,6,1,2,3] # setting column's order
df = df[[df.columns[i] for i in order]]
#------------------------------------------------------------------------------------------------------

# Ahora transformaremos los datos de posición y tamaño de plataforma a "Pool Diameter"
# como unidad de medida. Después de un analisis de un ensayo "vuelta olímpica, determinamos que
# el Diametro de la piscina son 280x2 = 560 unidades virturales asi que:
def normalizar_a_PD(uvirtuales):
    return (uvirtuales/560)



df.to_excel('MergedDataFrame.xlsx')