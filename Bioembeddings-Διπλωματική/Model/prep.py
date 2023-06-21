
# coding: utf-8

# In[10]:


import argparse
from pathlib import Path
from encoder.preprocess import Dataset_list

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Preprocesses")
 parser.add_argument("datasets_root", type=Path, help=        "Path")
 args = parser.parse_args()
 
 Dataset_list(args.datasets_root,13)

 

