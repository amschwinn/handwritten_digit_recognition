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
from skimage.transform import resize, downscale_local_mean
from skimage.filters import threshold_otsu
from scipy import ndimage 
from skimage.morphology import dilation, disk
import pickle

#%%
uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAMgAAADICAYAAACtWK6eAAAFkUlEQVR4nO3dS3LbOhAF0N7/4ryNzPImnjiJnTdQ4tiWSAkgQPzOqdLUJlF92QC/EQAAAAAAAAAAAAAAAAAAAAAArOMtIn7v/N7abRq0tReMrz9Yws+43zV0EpbzGumh0EVYwtFgCAjTKhUOAWEqP6JsOASEaeQswi3SmV6JhbjuwXR+Rr1glOwc/538/2CY6ZSOxalKB+MtLp2ohu+Z2wNJakylaoXiI+seqqq1+D5L79vHoGqsL1pMXwSE4moE43dcLiCeTUAoptYp25aL3kdO7/a2zXRolFO2OXQRDqkRjjPOUKXIOd0rJIurEYzXU/cgnZDwkFmnU/d4opG7SnaO3jvGLboIm0qFo7c1RgoB4aZVplH3CAhXjnaOWcIRkTcWTG7V6dQWASEijneOmRmHxR0Jx0xTqi0CsjDhuE9AFmZatS/l1USrHDCWonPsc8BYnHBsS51+Mhnn+fcZm8UpgG2pD4Wt1FmXIRzbTK9wdNyQ+lLtFs/QcwJHx9tMrYgIAbkldWo14/1n/CEgn+V8r4SJKYLPUsNhejU5AfnHNSGuKISLnKmVM1cLMJW4MLXiJlOK9DGYdRy4YfUjZ866w9RqIasfPVc/QHBHTkBmKRJnrbgrJyCzTDNW3GcSrfo8+kr7ykG5ARl12rHCPlLYKp3EG9s5JDcooxSR7sFhuZ2k99u+c24nGSX4nOjIdKtXufsENx0JSW8fypkt7HTi6Musn87f5E9m7IR0Jmfe3sMcfsRtZmBHCu7Moivx2ThIlvoStRZBKbF9bichW8kv3pYMSqntMrXisJIhKTGd6TGwLK50SHIvMAoH3SodktQi7aWDwaanaBMSXYOhvMZ5QREOhlV62vW3mJ8L/31oqkZQdA6m00tQoGutgqJjMIyzQwLDKXU/l87B1HQPuKPWlKv35+IhSa2gvEXEtxP3A6qquYjv7Rl5yPYcdRfvugpDq72A11UY1lnh2Oswf3+vEfFSd3fhca3D8Uh4nCGjidK3ywsLU2ld8MJCt3I/R9DLncJftwuKyS3Ej35Gf2ERFA6r8bb13tYypl9kKx2Oj76FrsLAcoo390jcY1eBXTW7x5beugrc1MtHNF/i0l0+Xj0XEpobqZDOOEMG71LXAz3N12uGBSJi3HB8VSMsLO5bzFkwJYPCwlIKqefuscVrUcmW+iHQkZ/6KzH9YjGrFkhuQEbsoGRKPZrO9iisLsIuhSEg7DC1uBAQrvwKRfGRseCTlPXHj0bbeDYB4V1KQFZhPHhn7XHt0THxJOICBOSaMeHdo8Xwq9UGNmChzjuFcC1lXfbUaBs5iYBcS/kM3Wx3FfCFgNxmHUJECMgW6xAiQhFscX2IiFAEW1KezXc9ZGICcttTWIcQArLHOgQFsMM6BAWwI+V6CJNSAPuMz+IUwD7jszgFsM/4LE4B7DM+i1MA+4zP4hTAPuOzOAWwz/gsTgFsew7jszwFsM2VdBTAjkfHZtXxWYICuC31U3RMSgHclhIOt7tP7NEiWOm1P6mfg3hus5mcwVHys5RgrNpdlyIg/+SEw2t/JudU5kVOOGYfE0JAIvLD4WUNC0j5gM5sBZHyxOBKBwy+WHEdcvQz0DONBXekFMYMX5k6Gg7dYzGpBTOq1KvjOgcRkT4XH61Ijq41Rj8wUMCsxVJiOjXS/lJJTiH12kleolwwhIN3oxdQyVD0fACgkR8xXiGVWHj3tk90bIQuknJxM+c32wVRCsqdptQ64v7687dLT596CDqDalFcZwfBlIpDWhSpYDCU1sVb+2etwWGtpjw1fx5yoqjWBV3qZzpFFaN3kZVeOEEjrYs89WcaxelaF/29nykUzbUOwa1QvFTdY0jU8mLeW1hTMIhaQREEAAAAAAAAAAAAAAAAAAAAAAAAAOb0PzyT9x8qCzdbAAAAAElFTkSuQmCC"


img = Image.open(io.BytesIO(base64.b64decode(uri.split(',')[1])))
#From image to np matrix. Ravel to 1D
img = np.array(img)[:,:,3].ravel()
#From 1D column to 1D row
img = np.reshape(img, (1, len(img)))
digit_imgs = img

digit_imgs = np.multiply(digit_imgs,1.0/255.0)

digit_imgs = digit_imgs.reshape(int(sqrt(digit_imgs.shape[1])),int(sqrt(digit_imgs.shape[1])))
#%%
#Select image
#img = Image.fromarray(digit_imgs[0,0,:,:])
img = digit_imgs

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
#%%    
#Extract just the digit
if (y_min+bound)>200 or (x_min+bound)>200:
    img = img[y_min:y_max,x_min:x_max]
else:
    img = img[y_min:(y_min+bound),x_min:(x_min+bound)]
    
#%%
#Convert image size
check = resize(np.array(img),(20,20),mode='reflect')
check2 = Image.fromarray(img).resize((20,20), Image.ANTIALIAS)
check3 = downscale_local_mean(img,(6,6))

#%%
img = check2
#%%
def padwith(vector, pad_width, iaxis, kwargs):
    vector[:pad_width[0]] = 0
    vector[-pad_width[1]:] = 0
    return vector

img = np.lib.pad(img,4,padwith)

#%%
thresh = threshold_otsu(img)

img_binary = img > thresh
img_binary = ndimage.binary_fill_holes(img_binary).astype(bool)

img_dilated = dilation(img_binary, disk(1))
#img_new = img_dilated - img_binary
img_new = np.bitwise_xor(img_dilated, img_binary)
    


#%%
#Test prediction
import freeman_code as fc
import knn

freeman_train = pickle.load(open('processed_data/freeman_train3_2k.sav','r'))
freeman_labels = pickle.load(open('processed_data/freeman_labels3_2k.sav','r'))

img_code = fc.freeman_code(img_new)
pred = knn.knn(img_code, freeman_train, freeman_labels, 1)
print(pred)
