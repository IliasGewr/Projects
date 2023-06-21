
# coding: utf-8

# In[5]:


from pathlib import Path
import os
import numpy as np
import pandas as pd
import glob

class Dataset_merge():
  
  def __init__(self,root):
   folder_path=Path(root)
   os.chdir(folder_path)
   file_extension = '.csv'
   all_filenames = [i for i in glob.glob(f"*{file_extension}")]
   combined_csv_data = pd.concat([pd.read_csv(f,skiprows=1,header=None) for f in all_filenames],axis=0,ignore_index=True)
   combined_csv_data.to_csv('combined_csv_data.csv')
