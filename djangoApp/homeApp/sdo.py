import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path

def apiSdo(path):
    # read csv file    
    df = pd.read_csv(os.path.join(path,'sdo.csv'))                
    new_df = pd.DataFrame()
    new_df['saturated_dissolved_oxygen(mg/L)'] = np.exp(-139.3441 + ((1.575701e5)/(df['water_temperature']+273)) - ((6.642308e7)/(df['water_temperature']+273)**2) + ((1.243800e10)/(df['water_temperature']+273)**3) - ((8.621949e11)/(df['water_temperature']+273)**4))

    new_df.index = df.index    

    # export the new_df into saturated_DO.csv
    new_df.to_csv(os.path.join(path,'saturated_DO.csv'), index=False)
    return 