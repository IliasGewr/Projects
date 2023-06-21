from pathlib import Path
import os
import numpy as np
import pandas as pd

class Dataset_list():
  
  def __init__(self,root,length):
    folder_path=Path(root)
    outputfile=folder_path.joinpath("_sources.txt")  #Path for _sources.txt file
    exclude= ["_sources.txt"]   #file types to exclude
    pathsep= "/"

#Writes in a txt file the dataset folder files.
        
        
    with open(outputfile, "w") as txtfile:
     for path,dirs,files in os.walk(folder_path):
       
      for fn in sorted(files):
        
               
       if not any(x in fn for x in exclude) :
        
        filename = os.path.splitext(fn)[0]

        
        txtfile.write("%s\n" % filename)

    txtfile.close()
    
#Creates a folder for each dataset, containing npy files of sub datasets(Utterances) of it.
    
    
    a_file = open(outputfile)
    lines = a_file. readlines()
    outputfile2=folder_path.joinpath("_sources.txt")

    for line in lines:
 
     speaker_out_dir = folder_path.joinpath(line.strip()+"1")
     
     speaker_out_dir.mkdir()
     
     outputfile2=speaker_out_dir.joinpath("_sources.txt")
     
     z=folder_path.joinpath(line.strip()+".csv")

#Utterances creation    
    
     df=pd.read_csv(z)
     for i in range(0,len(df),length):
      dat=np.empty(shape=(1,df.shape[1]),dtype='float64')
      train_name="batch_" + str(i) + ".npy"
      x=i+length
      if x>len(df) :
       for w in range(i,len(df)):
        x=np.array(df.iloc[[w]])
        dat=np.concatenate((dat, x))
        np.save(os.path.join(speaker_out_dir, train_name), dat)
      else:
       for w in range(i,i+length):
        x=np.array(df.iloc[[w]])
        dat=np.concatenate((dat, x))
        np.save(os.path.join(speaker_out_dir, train_name), dat)
    
#Creates a txt files that contains Utterances names and paths     

      with open(outputfile2, "w") as txtfile:
       for path,dirs,files in os.walk(speaker_out_dir):
       
        for fn in sorted(files):
        
               
         if not any(x in fn for x in exclude) :
        
          filename = os.path.splitext(fn)[0]
          filename=filename + ".npy"+","+str(z)

        
          txtfile.write("%s\n" % filename)
          

      txtfile.close()
x=2
