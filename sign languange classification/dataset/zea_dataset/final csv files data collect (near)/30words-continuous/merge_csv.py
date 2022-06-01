"""import pandas as pd
import glob
import os
import csv

files = os.path.join("C:\\Users\\markc\\OneDrive\\Desktop\\thesis_final\\python\\*.csv")
files = glob.glob(files)

df = pd.concat(map(pd.read_csv,files), ignore_index=True)
print(df)

with open('wlasl.csv', mode='a', newline='') as f:
    csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    csv_writer.writerow(df)"""
                	
import os
import pandas as pd

master_df = pd.DataFrame()

for file in os.listdir(os.getcwd()):
    if file.endswith('.csv'):
        master_df = master_df.append(pd.read_csv(file))

master_df.to_csv('all_csv03.csv', index=False)