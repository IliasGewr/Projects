#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from pathlib import Path
from encoder.pca import pca_rand

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="PCA")
 parser.add_argument("datasets_root", type=Path, help="Path")
 args = parser.parse_args()
 
 pca_rand(args.datasets_root)

