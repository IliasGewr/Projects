
# coding: utf-8

# In[1]:


import argparse
from pathlib import Path
from encoder.merge import Dataset_merge

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Dataset merge")
 parser.add_argument("datasets_root", type=Path, help="Path")
 args = parser.parse_args()
 
 Dataset_merge(args.datasets_root)

