#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from sklearn.utils.extmath import randomized_svd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import pandas as pd

class pca_rand():
  
  def __init__(self,root):
   folder_path=Path(root)
   data = pd.read_csv(folder_path)
    
   print("Data loaded")
   
   data=data.to_numpy()

   print("Data to numpy")
   
   data=data[:,6:]
   print(data.shape)
   
   scaler = StandardScaler()
   data = scaler.fit_transform(data)
   
   print("Scaling finished")

   U, s, V = randomized_svd(data, 500)
    
   print("PCA finished")

   pd.DataFrame(V).to_csv("pca_matrix.csv")
   

