# ---------------------------------------------------------
# Lab ONCE - Diciembre 2022
# Fondecyt 11200469
# Hayo Breinbauer
# ---------------------------------------------------------
# Procesar formulario SRQ para tesis Rosario Garrido
# ---------------------------------------------------------


import pandas as pd     #Base de datos
import numpy as np      # Libreria de calculos científicos
import seaborn as sns   #Estetica de gráficos
import matplotlib.pyplot as plt    #Graficos


data = pd.read_csv('SRQ_2022_12_07.csv', index_col=0)
# Aqui leo el CSV que descargué de Google Forms. Ojo que le cambié el nombre para dejar marcada
# la fecha en que lo bajé.
data.columns.values[0]='PXX'
PXX = data.pop('PXX')
data = data.replace(regex=[r'\D+'], value="").astype(int)

data.insert(0, 'PXX', PXX)

#Aqui eliminé todos los caracters no númericos de la base de datos. Me quedo solo con los numeros.

print(data)