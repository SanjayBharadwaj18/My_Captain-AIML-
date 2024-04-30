#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip install fastai')
#importing libraries
from fastai import *
from fastai.vision import *
from fastai.metrics import error_rate
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# **Point to be Noted: Number of elements in a list of path is same as number of classes you have**

# In[ ]:


x  = 'C:\Users\Sanjay Bharadwaj U\Desktop\My cap\AIML(Assignment-5)\seg_pred'
path = Path(x)
path.ls()


# In[ ]:


np.random.seed(40)
data = ImageDataBunch.from_folder(path, train = '.', valid_pct=0.2,
                                  ds_tfms=get_transforms(), size=224,
                                  num_workers=4).normalize(imagenet_stats)


# In[ ]:


data.show_batch(rows=3, figsize=(7,6),recompute_scale_factor=True)


# In[ ]:


data


# In[ ]:


print(data.classes)
len(data.classes)
data.c


# In[ ]:


learn = cnn_learner(data, models.resnet18, metrics=[accuracy], model_dir = Path('../kaggle/working'),path = Path("."))


# In[ ]:


learn.lr_find()
learn.recorder.plot(suggestions=True)


# In[ ]:


lr1 = 1e-3
lr2 = 1e-1
learn.fit_one_cycle(40,slice(lr1,lr2))


# In[ ]:


learn.unfreeze()
learn.fit_one_cycle(20,slice(1e-4,1e-3))


# In[ ]:


learn.recorder.plot_losses()


# Model performance can be validated in different ways. One of the popular methods is using the confusion matrix. Diagonal values of the matrix indicate correct predictions for each class, whereas other cell values indicate a number of wrong predictions.

# In[ ]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[ ]:


interp.plot_top_losses(6,figsize = (25,5))


# In[ ]:


img = open_image('/kaggle/input/intel-image-classification/seg_test/seg_test/glacier/21982.jpg')
print(learn.predict(img)[0])


# In[ ]:


img = open_image('/kaggle/input/intel-image-classification/seg_test/seg_test/glacier/21982.jpg')
print(learn.predict(img)[0])


# In[ ]:


learn.export(file = Path("/kaggle/working/export.pkl"))
learn.model_dir = "/kaggle/working"
learn.save("stage-1",return_path=True)


# 
