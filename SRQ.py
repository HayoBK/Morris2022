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

print(data)