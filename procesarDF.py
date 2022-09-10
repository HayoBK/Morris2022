import pandas as pd     #Base de datos
import numpy as np

m_df = pd.read_csv('MergedDataFrame.csv', index_col=0)
for col in m_df.columns:
    print(col)
FirstRow=True
for row in m_df.itertuples():
    if FirstRow:
        AssesedTrial = row.Trial_Unique_ID
        print('First! es ', AssesedTrial)
        FirstRow = False
    if AssesedTrial != row.Trial_Unique_ID:
        FirstRow = True
    print(AssesedTrial, row.Trial_Unique_ID, FirstRow)

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