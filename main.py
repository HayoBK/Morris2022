# El desafío es poder procesar lo que emite Simian...

import json

import pandas as pd
from flatten_json import flatten
import pandas
with open("Navi.json", "r") as read_file:
    data = json.load(read_file)

#print(data)
#print(json.dumps(data, indent=4))
#flat_data= flatten(data)
#print(flat_data)
for key in flatten(data).keys():
    print(key)

df = pd.json_normalize(data, record_path=['motionTrackDataPerTrials'], meta=['platformExists','platformPosition.x','platformPosition.y'], errors='ignore')


#df = pd.json_normalize(data)
pd.set_option('display.max_columns', None)
print(df)