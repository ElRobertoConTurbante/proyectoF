import cv2
import tensorflow as tf
from matplotlib import pyplot as plt
from keras import backend as K
from glob import glob
import numpy as np
import os
import streamlit as st
from PIL import Image

#Read data 

import pandas as pd
df = pd.read_csv('train.csv')


# Working with the dataframe. 
df.rename(columns = {'class':'class_name'}, inplace = True)

# Creating new columns called case, day and slice to store caseid, day9d, slideid
df["case"] = df["id"].apply(lambda x: int(x.split("_")[0].replace("case", "")))
df["day"] = df["id"].apply(lambda x: int(x.split("_")[1].replace("day", "")))
df["slice"] = df["id"].apply(lambda x: x.split("_")[3])

# Extracting images from Train folder
TRAIN_DIR="/Users/carlosnino/Documents/random/AML/proyectoF/train"
all_train_images = glob(os.path.join(TRAIN_DIR, "**", "*.png"), recursive=True)
x = all_train_images[0].rsplit("/", 4)[0] 

path_partial_list = []
for i in range(0, df.shape[0]):
    path_partial_list.append(os.path.join(x,
                          "case"+str(df["case"].values[i]),
                          "case"+str(df["case"].values[i])+"_"+ "day"+str(df["day"].values[i]),
                          "scans",
                          "slice_"+str(df["slice"].values[i])))
df["path_partial"] = path_partial_list

path_partial_list = []
for i in range(0, len(all_train_images)):
    path_partial_list.append(str(all_train_images[i].rsplit("_",4)[0]))
    
tmp_df = pd.DataFrame()
tmp_df['path_partial'] = path_partial_list
tmp_df['path'] = all_train_images

# Adding the path to images to the dataframe 
df = df.merge(tmp_df, on="path_partial").drop(columns=["path_partial"])

# Creating new columns height and width from the path details of images
df["width"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))
df["height"] = df["path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))

# Deleting redundant columns
del x,path_partial_list,tmp_df


def rle_decode(mask_rle, shape):
    '''mask_rle: run-length as string formated (start length)
    shape: (height,width) of array to return 
    Returns numpy array, 1 - mask, 0 - background
    '''
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0:][::2], s[1:][::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0]*shape[1], dtype=np.float32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape)  # Needed to align to RLE direction


df_train = pd.DataFrame({'id':df['id'][::3]})

df_train['large_bowel'] = df['segmentation'][::3].values
df_train['small_bowel'] = df['segmentation'][1::3].values
df_train['stomach'] = df['segmentation'][2::3].values

df_train['path'] = df['path'][::3].values
df_train['case'] = df['case'][::3].values
df_train['day'] = df['day'][::3].values
df_train['slice'] = df['slice'][::3].values
df_train['width'] = df['width'][::3].values
df_train['height'] = df['height'][::3].values

df_train.reset_index(inplace=True,drop=True)
df_train = df_train.dropna()
df_train.reset_index(inplace=True,drop=True)


im_large = df_train['large_bowel'][0]
mask_large = rle_decode(im_large, (266,266,1))

num_0_large = list(mask_large[:][:].flatten()).count(0)
num_1_large = list(mask_large[:][:].flatten()).count(1)


weight_for_0_large = (1.0 / num_0_large) * ((num_0_large+num_1_large) / 2.0)
weight_for_1_large = (1.0 / num_1_large) * ((num_0_large+num_1_large) / 2.0)

class_weights_large = {0:weight_for_0_large, 1:weight_for_1_large}

im_small = df_train['small_bowel'][0]
mask_small = rle_decode(im_small, (266,266,1))

num_0_small = list(mask_small[:][:].flatten()).count(0)
num_1_small = list(mask_small[:][:].flatten()).count(1)


weight_for_0_small = (1.0 / num_0_small) * ((num_0_small+num_1_small) / 2.0)
weight_for_1_small = (1.0 / num_1_small) * ((num_0_small+num_1_small) / 2.0)

class_weights_small = {0:weight_for_0_small, 1:weight_for_1_small}

im_stomach = df_train['stomach'][0]
mask_stomach = rle_decode(im_stomach, (266,266,1))

num_0_stomach = list(mask_stomach[:][:].flatten()).count(0)
num_1_stomach = list(mask_stomach[:][:].flatten()).count(1)


weight_for_0_stomach = (1.0 / num_0_stomach) * ((num_0_stomach+num_1_stomach) / 2.0)
weight_for_1_stomach = (1.0 / num_1_stomach) * ((num_0_stomach+num_1_stomach) / 2.0)

class_weights_stomach = {0:weight_for_0_stomach, 1:weight_for_1_stomach}

def custom_binary_loss(y_true, y_pred):
  y_true = K.flatten(y_true)
  y_pred = K.flatten(y_pred)
  y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    
  term_0 = 1.4 * weight_for_0_large*((1 - y_true) * K.log(1 - y_pred + K.epsilon()))  # Cancels out when target is 1 
  term_1 = 0.55 * weight_for_1_large*(y_true * K.log(y_pred + K.epsilon())) # Cancels out when target is 0

  return -K.mean(term_0 + term_1, axis=-1)


best_model = tf.keras.models.load_model('best_model_AML.h5', custom_objects={'custom_binary_loss': custom_binary_loss})


img_path = df_train['path'][1]
img = cv2.imread(img_path, cv2.IMREAD_ANYDEPTH)
img = (img - img.min())/(img.max() - img.min())*255.0 
img = cv2.resize(img, (256, 256))
img = np.expand_dims(img, axis=-1)
img = img.astype(np.float32) / 255.
pred = best_model.predict(np.array([img]))




def color_class(pred):
  classes = {'large bowel':0.3, 'small bowel':0.5, 'stomach':0.9}
  threshold = 0.95
  t = 0.94999
  pixels_to_change_large = np.where(pred[0][0] > threshold)
  pixels_to_change1_large = np.where(pred[0][0] < t)

  pixels_to_change_small = np.where(pred[1][0] > threshold)
  pixels_to_change1_small = np.where(pred[1][0] < t)

  pixels_to_change_stomach = np.where(pred[2][0] > threshold)
  pixels_to_change1_stomach = np.where(pred[2][0] < t)
  # Modify identified pixels
  pred[0][0][pixels_to_change_large] = classes['large bowel']
  pred[0][0][pixels_to_change1_large] = 0
  pred[1][0][pixels_to_change_small] = classes['small bowel']
  pred[1][0][pixels_to_change1_small] = 0
  pred[2][0][pixels_to_change_stomach] = classes['stomach']
  pred[2][0][pixels_to_change1_stomach] = 0

  tot_mask = pred[0][0] + pred[1][0] + pred[2][0]

  return tot_mask

mask_pred = color_class(pred)

plt.figure(figsize=(5*5, 7))
plt.imshow(img, cmap='bone', alpha=1)
plt.imshow(mask_pred, alpha=0.45, cmap='hot')
plt.axis('off')


def main():
   
   st.title('Optimización del tratamiento de cáncer GI')
   st.markdown('Elige una imagen de validación para probar el modelo.')

   option = st.selectbox('Imagen:',('81', '82', '83', '84'))
   
   path = 'slice_00'+ option +'_266_266_1.50_1.50.png'


   st.write('Imagen seleccionada:', option)


   


   img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
   img = (img - img.min())/(img.max() - img.min())*255.0 
   img = cv2.resize(img, (256, 256))
   img = np.expand_dims(img, axis=-1)
   img = img.astype(np.float32) / 255.
   fig, ax = plt.subplots()
   ax.imshow(img)

   st.write('Imagen seleccionada')
   st.pyplot(fig)

   pred = best_model.predict(np.array([img]))
   mask_pred = color_class(pred)

   fig1, ax1 = plt.subplots()
   ax1.imshow(img, cmap='bone', alpha=1)
   st.write('Imagen con predicción:')
   ax1.imshow(mask_pred, alpha=0.45, cmap='hot')
   st.pyplot(fig1)




if __name__ == '__main__':
    main()