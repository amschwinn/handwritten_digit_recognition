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
from math import sqrt

#%%
def load_user_examples():
    #Connect to mysql db
    engine = create_engine("mysql://mldm_gangster:$aint3tienne@ml-digit-recognition.cnpjv4qug6jj.us-east-2.rds.amazonaws.com/digits")
    #load the table into a data frame
    digits = pd.read_sql_table("user_digits",engine)
    
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

def padwith(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector


def format_user_examples(digit_imgs):
    #load all  images
    digit_imgs = digit_imgs.values
    digit_imgs = np.multiply(digit_imgs,1.0/255.0)
    digit_imgs = digit_imgs.reshape(int(digit_imgs.shape[0]),1,
                    int(sqrt(digit_imgs.shape[1])),int(sqrt(digit_imgs.shape[1])))
  
    #Iterate through images image
    for i in xrange(len(digit_imgs)):
        #img = Image.fromarray(digit_imgs[0,0,:,:])
        img = digit_imgs[i,0,:,:]
        
        #Get just the digit
        x_min = np.min((np.sum(img,axis=0) != 0).nonzero())
        x_max = np.max((np.sum(img,axis=0) != 0).nonzero())
        y_min = np.min((np.sum(img,axis=1) != 0).nonzero())
        y_max = np.max((np.sum(img,axis=1) != 0).nonzero())
        
        #Get the bound box size
        if (x_max - x_min) > (y_max - y_min):
            bound = (x_max - x_min)
        else:
            bound = (y_max - y_min)
   
        #Extract just the digit
        if (y_min+bound)>200 or (x_min+bound)>200:
            img = img[y_min:y_max,x_min:x_max]
        else:
            img = img[y_min:(y_min+bound),x_min:(x_min+bound)]
            

        #Convert image size
        img = Image.fromarray(img).resize((20,20), Image.ANTIALIAS)
    
    
        img = np.lib.pad(img,4,padwith) 

#%%
digit_imgs, labels = load_user_examples()

