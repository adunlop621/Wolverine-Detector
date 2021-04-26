#!/usr/bin/env python
# coding: utf-8

# In[74]:


#hide
# !pip install -Uqq fastbook
import fastbook
fastbook.setup_book()


# In[75]:


#hide
from fastbook import *
from fastai.vision.widgets import *


# In[1]:


get_ipython().system('pip install voila')
get_ipython().system('jupyter serverextension enable --sys-prefix voila')


# In[76]:


#key = os.environ.get('AZURE_SEARCH_KEY', '68d26e8ca41f4b2ab2d32f96e9d58f3c')


# In[77]:


#hugh_jackman_pics = search_images_bing(key, "hugh jackman")
#ims = hugh_jackman_pics.attrgot('contentUrl')
#len(ims)


# In[78]:


#dest = 'images/hugh_jackman.jpg'
#download_url(ims[0], dest)
#path = Path("Huge Jacked Man")


# In[79]:


#dest = (path/"Wolverine")
#dest.mkdir(exist_ok=True)
#wolverine_pics = search_images_bing(key, "hugh jackman wolverine")
#download_images(dest, urls = wolverine_pics.attrgot('contentUrl'))


# In[80]:


#dest = (path/"Hugh Jackman")
#dest.mkdir(exist_ok=True)
#hugh_jackman_pics = search_images_bing(key, "hugh jackman")
#download_images(dest, urls = hugh_jackman_pics.attrgot('contentUrl'))


# In[81]:


path = Path("Huge Jacked Man")
fns = get_image_files(path)
failed = verify_images(fns)


# In[82]:


failed.map(Path.unlink);


# In[83]:


huge_jacked_man = DataBlock(
    blocks=(ImageBlock, CategoryBlock), 
    get_items=get_image_files, 
    splitter=RandomSplitter(valid_pct=0.2, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[84]:


dls = huge_jacked_man.dataloaders(path)


# In[85]:


dls.valid.show_batch(max_n=8, nrows=3)


# In[86]:


huge_jacked_man = huge_jacked_man.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms(mult=2))
dls = huge_jacked_man.dataloaders(path)


# In[87]:


learn = cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(8)


# In[88]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[89]:


interp.plot_top_losses(5, nrows=1)


# In[57]:


#hide_output
cleaner = ImageClassifierCleaner(learn)
cleaner


# In[50]:


for idx in cleaner.delete(): cleaner.fns[idx].unlink()


# In[111]:


#import shutil
#shutil.rmtree(path/'Wolverine', ignore_errors=True)


# In[90]:


learn.export()


# In[91]:


path = Path()
path.ls(file_exts='.pkl')


# In[92]:


learn_inf = load_learner(path/'export.pkl')


# In[93]:


learn_inf.dls.vocab


# In[94]:


#hide_output
btn_upload = widgets.FileUpload()


# In[101]:


img = PILImage.create(btn_upload.data[-1])


# In[96]:


#hide_output
btn_run = widgets.Button(description='Classify')


# In[102]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[116]:


btn_upload = widgets.FileUpload()


# In[117]:


#hide_output
VBox([widgets.Label('Select your Huge Jacked Man!'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[ ]:




