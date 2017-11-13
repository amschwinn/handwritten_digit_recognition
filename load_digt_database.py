# -*- coding: utf-8 -*-
"""
Load training examples from
user input hand written dataset

By: Austin Schwinn

November 11, 2017

Notes: Have to pip install Pillow,
sqlalchemy, mysql-python

"""
import numpy as np
import pandas as pd
from PIL import Image
import io 
import base64
from sqlalchemy import create_engine

#%%
def load_user_examples():
    #Connect to mysql db
    engine = create_engine("mysql://mldm_gangster:$aint3tienne@ml-digit-project.cnpjv4qug6jj.us-east-2.rds.amazonaws.com/digits")
    #load the table into a data frame
    digits = pd.read_sql_table("user_digits",engine)
    
    #%%
    digit_imgs = []
    labels = []
    #Convert image uri to 1D pixel array
    for index, value in digits.iterrows():
        #Get image uri
        uri = value.image
        #Convert from uri to PIL image
        img = Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1])))
        #From image to np matrix. Ravel to 1D
        img = np.array(img)[:,:,3].ravel()
        #From 1D column to 1D row
        img = np.reshape(img, (1, len(img)))
        digit_imgs = digit_imgs + [img[0]]
        labels = labels + [value.label]
    
    #conver to dataframes for easier use    
    digit_imgs = pd.DataFrame.from_records(digit_imgs)
    labels = pd.DataFrame(np.array(labels))

    return digit_imgs, labels

