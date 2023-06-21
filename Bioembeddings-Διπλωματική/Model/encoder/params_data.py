
# coding: utf-8

# In[ ]:


## Model parameters

layer1_size=1024
layer2_size=512
layer3_size=54

##Data parameters
studies_per_batch=5
samples_per_study=125
batch_size=studies_per_batch*samples_per_study
input_shape=999

learning_rate_init = 1e-5

partials_n_frames = 1

