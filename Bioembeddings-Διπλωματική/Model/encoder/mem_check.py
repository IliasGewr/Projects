#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.utils.extmath import randomized_svd


class pca():
  
  def __init__(self,root):
   folder_path=Path(root)
   data = pd.read_csv(folder_path)
   data=data.to_numpy()
   pca = PCA(n_components=300,svd_solver='auto')
   pca.fit(data)


class pca_incr():
  
  def __init__(self,root):
   folder_path=Path(root)
   data = pd.read_csv(folder_path)
   data=data.to_numpy()
   transformer = IncrementalPCA(n_components=300, batch_size=300)
   transformer.fit(data)


class pca_rand():
  
  def __init__(self,root):
   folder_path=Path(root)
   data = pd.read_csv(folder_path)
   data=data.to_numpy()
   U, s, V = randomized_svd(data, 300)

