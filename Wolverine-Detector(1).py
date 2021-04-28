#!/usr/bin/env python
# coding: utf-8

# In[2]:


# !pip install -Uqq fastbook
import fastbook
from fastbook import *
from fastai.vision.widgets import *
path = Path()
learn_inf = load_learner(path/'export.pkl', cpu=True)
btn_upload = widgets.FileUpload()
out_pl = widgets.Output()
lbl_pred = widgets.Label()

def on_click(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
    
btn_upload.observe(on_click,names=['data'])
desLbl = widgets.Label('Upload a picture of Wolverine or Hugh Jackman!')
display(VBox([desLbl, 
      btn_upload, out_pl, lbl_pred]))


# In[ ]:




