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



home= str(Path.home()) # Obtener el directorio raiz en cada computador distinto
BaseDir=home+"/OneDrive/2-Casper/00-CurrentResearch/001-FONDECYT_11200469/002-LUCIEN/SUJETOS/"

FileName = BaseDir + 'P06/LSL_LAB/ses-NI/eeg/sub-P006_ses-NI_task-Default_run-001_eeg.xdf'

data, header = pyxdf.load_xdf(FileName)

print('hola')