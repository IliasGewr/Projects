#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import argparse
from pathlib import Path
from encoder.mem_check import pca, pca_incr, pca_rand

if __name__ == "__main__":
 parser = argparse.ArgumentParser(description="Memory check")
 parser.add_argument("datasets_root", type=Path, help="Path")
 args = parser.parse_args()
 
 
 pca(args.datasets_root)
 pca_incr(args.datasets_root)
 pca_rand(args.datasets_root)

