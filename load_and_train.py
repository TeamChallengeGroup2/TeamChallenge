#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import cv2
from scipy.ndimage import label as ndlabel

import SimpleITK as sitk
import matplotlib.pyplot as plt
import PIL
import keras
import pickle
import tensorflow as tf

from keras import optimizers
from keras.layers import Dropout, Lambda, average, ZeroPadding2D, Cropping2D
import random

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# In[30]:


def mvn(tensor):
    '''Performs per-channel spatial mean-variance normalization.'''
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)
    
    return mvn

def tversky_loss_2d(Y_gt, Y_pred):
    smooth = 1e-5
    alpha = 0.5
    beta = 0.5
    ones = tf.ones(tf.shape(Y_gt))
    p0 = Y_pred
    p1 = ones - Y_pred
    g0 = Y_gt
    g1 = ones - Y_gt
    num = tf.reduce_sum(p0 * g0, axis=[1, 2])
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=[1, 2]) +           beta * tf.reduce_sum(p1 * g0, axis=[1, 2]) + smooth
    tversky = tf.reduce_sum(num / den, axis=1)
    loss = tf.reduce_mean(1 - tversky)
    return loss

def dice_coef(y_true, y_pred, smooth=0.1):
    '''Average dice coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)


def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)


def jaccard_coef(y_true, y_pred, smooth=0.1):
    '''Average jaccard coefficient per batch.'''
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)


def fcn_model(input_shape, num_classes, weights=None):
    ''' "Skip" FCN architecture similar to Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = tversky_loss_2d
        activation = 'sigmoid'
    else:
        loss = 'categorical_crossentropy'
        activation = 'softmax'

    kwargs = dict(
        kernel_size=3,
        strides=1,
        activation='relu',
        padding='same',
        use_bias=True,
        kernel_initializer='glorot_uniform',
        bias_initializer='zeros',
        bias_regularizer=None,
        activity_regularizer=None,
        kernel_constraint=None,
        bias_constraint=None,
        trainable=True,
    )
    
    data = Input(shape=input_shape, dtype='float', name='data')
    mvn0 = Lambda(mvn, name='mvn0')(data)
    pad = ZeroPadding2D(padding=2, name='pad')(mvn0)

    conv1 = Conv2D(filters=64, name='conv1', **kwargs)(pad)
    mvn1 = Lambda(mvn, name='mvn1')(conv1)
    
    conv2 = Conv2D(filters=64, name='conv2', **kwargs)(mvn1)
    mvn2 = Lambda(mvn, name='mvn2')(conv2)

    conv3 = Conv2D(filters=64, name='conv3', **kwargs)(mvn2)
    mvn3 = Lambda(mvn, name='mvn3')(conv3)
    pool1 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool1')(mvn3)

    
    conv4 = Conv2D(filters=128, name='conv4', **kwargs)(pool1)
    mvn4 = Lambda(mvn, name='mvn4')(conv4)

    conv5 = Conv2D(filters=128, name='conv5', **kwargs)(mvn4)
    mvn5 = Lambda(mvn, name='mvn5')(conv5)

    conv6 = Conv2D(filters=128, name='conv6', **kwargs)(mvn5)
    mvn6 = Lambda(mvn, name='mvn6')(conv6)
    
    upsample2 = Conv2DTranspose(filters=16, kernel_size=3,
                        strides=2, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample2')(mvn6)

    conv7 = Conv2D(filters=128, name='conv7', **kwargs)(upsample2)
    mvn7 = Lambda(mvn, name='mvn7')(conv7)
    
    
    pool2 = MaxPooling2D(pool_size=3, strides=2,
                    padding='valid', name='pool2')(mvn7)
    upsample1 = Conv2DTranspose(filters=16, kernel_size=3,
                        strides=2, activation=None, padding='same',
                        kernel_initializer='glorot_uniform', use_bias=False,
                        name='upsample1')(pool2)
    
    out = keras.layers.Dense(1, activation='sigmoid')(upsample1)
    
    model = Model(inputs=data, outputs=out)
    
    if weights is not None:
        model.load_weights(weights)
    sgd = optimizers.SGD(lr=0.01, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss=loss,
                  metrics=['accuracy', dice_coef, jaccard_coef])

    return model


# In[2]:
    
    
#This function resizes the images so all of them have the same spacing in the x-y plane

def respace(itk_image, new_spacing):
    spacing = itk_image.GetSpacing()
    size = itk_image.GetSize()
    new_size = (np.round(size*(spacing/np.array(new_spacing)))).astype(int).tolist()
    new_image = sitk.Resample(itk_image, new_size, sitk.Transform(),sitk.sitkNearestNeighbor,
                            itk_image.GetOrigin(), new_spacing, itk_image.GetDirection(), 0.0,
                            itk_image.GetPixelID())
    return new_image
    
    


def biggest_region_3D(array):
    if len(array.shape)==4:
        im_np=np.squeeze(array)
    else:
        im_np=array
    struct=np.full((3,3,3),1)
    c=0
    maxn=im_np.max()
    arr=np.where(im_np>=(maxn-0.2),1,0)
    lab, num_reg=ndlabel(arr,structure=struct)
    h=np.zeros(num_reg+1)
    for i in range(num_reg):
        z=np.where(lab==(i+1),1,0)
        h[i+1]=z.sum()
        if h[i+1]==h.max():
            c=i+1
    lab=np.where(lab==c,1,0)
    return lab


# In[3]:


def cropROI(arrayin, slicenr):
    #input: an array with indices [slicenr, imageX, imageY]
    # slicenr: averaging over this number of slices
    
    LVradius=20 #the radius used in frst, which should be the radius of the LV in the top image
    cropdiam=64 #the length in X and Y direction of the cropped image
    multi_mindist = []
    multi_circles = []
    sumdist1 = 0
    sumdist2 = 0
    
    for i in range(slicenr):
        topslice=arrayin[i,:,:]
        center=[topslice.shape[1]/2,topslice.shape[0]/2]        #find coordinates of center of image

        im = np.array((topslice/np.amax(topslice)) * 255, dtype = np.uint8)
    
        circles=cv2.HoughCircles(im, cv2.HOUGH_GRADIENT, 1, 70, param1=25, param2=20, minRadius=LVradius-14, maxRadius=LVradius+10)
        multi_circles.append(circles)

    #function to calculate distance from coordinates of HoughCircles to center of the image
        def calculateDistance(x1,y1,x2,y2):  
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)  
            return dist

    #calculate distances and add to list dist
        dist=[]
        for j in range(circles.shape[1]):
            d=calculateDistance(circles[0,j,0],circles[0,j,1],center[0],center[1])
            dist.append(d)

        mindist=np.argmin(dist)         #find index of minimal distance
        multi_mindist.append(mindist)
        sumdist1 += int(circles[0,mindist,1])
        sumdist2 += int(circles[0,mindist,0])

    sumdist1 = int(sumdist1/slicenr)
    sumdist2 = int(sumdist2/slicenr)
    
    #find the coordinates around which to crop
    if cropdiam>=sumdist1:
        cropcoor_x1=0
    else:
        cropcoor_x1=sumdist1-cropdiam
    cropcoor_x2=cropcoor_x1+2*cropdiam
    if cropcoor_x2>=topslice.shape[0]:
        cropcoor_x2=topslice.shape[0]
        cropcoor_x1=cropcoor_x2-2*cropdiam
        
    if cropdiam>=sumdist2:
        cropcoor_y1=0
    else:
        cropcoor_y1=sumdist2-cropdiam
    cropcoor_y2=cropcoor_y1+2*cropdiam
    if cropcoor_y2>=topslice.shape[1]:
        cropcoor_y2=topslice.shape[1]
        cropcoor_y1=cropcoor_y2-2*cropdiam
    
      
    return cropcoor_x1, cropcoor_x2, cropcoor_y1, cropcoor_y2


# In[4]:


#This function gives as output the end-diastolic and end-systolic 3D frames of 100 patients, including both the initial 
#and segmented images. If the output is the list Y:
#Y[0] = patient number
#Y[1] = number of slices per frame
#Y[2] = ED 3D original image AS AN ARRAY
#Y[3] = ED 3D ground truth AS AN ARRAY
#Y[4] = ES 3D original image AS AN ARRAY
#Y[5] = ES 3D ground truth AS AN ARRAY

def load3Ddata(path):
    #"path" is the path of the folder named "Data"
    l=[]
    for i, name in enumerate(os.listdir(path)):
        data = open('{}\{}\Info.cfg'.format(path,name), 'r')
        
    
        ED=data.readline()    #end-diastolic frame information
        for s in ED.split():
            if s.isdigit():   #end-diastolic frame number
            #reading the end-diastolic 3d images:
                if int(s)<=9:
                    im_EDframe= sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                    im_EDgt=sitk.ReadImage('{}\{}\{}_frame0{}_gt.nii.gz'.format(path,name,name,s))
                else:
                    im_EDframe= sitk.ReadImage('{}\{}\{}_frame{}.nii.gz'.format(path,name,name,s))
                    im_EDgt=sitk.ReadImage('{}\{}\{}_frame{}_gt.nii.gz'.format(path,name,name,s))
                
        ES=data.readline()    #end-systolic frame information
        for s in ES.split():
            if s.isdigit():   #end-systolic frame number
                #reading the end-systolic 3d images:
                if int(s)<=9:
                    im_ESframe= sitk.ReadImage('{}\{}\{}_frame0{}.nii.gz'.format(path,name,name,s))
                    im_ESgt=sitk.ReadImage('{}\{}\{}_frame0{}_gt.nii.gz'.format(path,name,name,s))
                else:
                    im_ESframe= sitk.ReadImage('{}\{}\{}_frame{}.nii.gz'.format(path,name,name,s))
                    im_ESgt=sitk.ReadImage('{}\{}\{}_frame{}_gt.nii.gz'.format(path,name,name,s))
                    
        
        #Definde the new spacing:
        z=im_EDframe.GetSpacing()[2]
        new_spacing=(1.0,1.0,z)
        
        if im_EDframe.GetSpacing()!=im_EDgt.GetSpacing() or im_ESframe.GetSpacing()!=im_ESgt.GetSpacing():
            im_EDgt.SetSpacing(im_EDframe.GetSpacing())
            im_ESgt.SetSpacing(im_ESframe.GetSpacing())
        
        im_EDframe=respace(im_EDframe,new_spacing)
        im_EDgt=respace(im_EDgt,new_spacing)
        im_ESframe=respace(im_ESframe,new_spacing)
        im_ESgt=respace(im_ESgt,new_spacing)
        
        
        
        #Converting the 3d images into 3 dimensional arrays:        
        arr_EDframe= sitk.GetArrayFromImage(im_EDframe)
        arr_EDgt= sitk.GetArrayFromImage(im_EDgt)
        arr_ESframe= sitk.GetArrayFromImage(im_ESframe)
        arr_ESgt= sitk.GetArrayFromImage(im_ESgt)
        
    
        NSlices=arr_EDframe.shape[0]
        av_n=int(NSlices*3/4)
        
        EDx1, EDx2, EDy1, EDy2=cropROI(arr_EDframe,av_n)

        arr_EDframe=arr_EDframe[:,EDx1:EDx2, EDy1:EDy2]
        arr_EDgt=arr_EDgt[:,EDx1:EDx2, EDy1:EDy2]
        arr_ESframe=arr_ESframe[:,EDx1:EDx2, EDy1:EDy2]
        arr_ESgt=arr_ESgt[:,EDx1:EDx2, EDy1:EDy2]
            
     
        #l=list with the patient number, the number of slices per frame, and the four 3D frames as arrays
        l.append([i+1, NSlices, arr_EDframe,arr_EDgt,arr_ESframe,arr_ESgt, z])
        
        
    return l


# In[16]:


def load_list_as_dataset(output):
    EDframes = []
    EDground = [] 
    ESframes = []
    ESground = []

    for i in range(len(output)):
        EDframes.append(output[i][2])
        EDground.append(output[i][3])
        ESframes.append(output[i][4])
        ESground.append(output[i][5])

    # Take the ES frames and ED frames 
    frames = ESframes+EDframes
    groundtruth = ESground+EDground

    frames=np.array(frames)
    groundtruth=np.array(groundtruth)
    frames = frames.astype('float32')
    groundtruth = groundtruth.astype('float32')
    groundtruth = np.where(groundtruth==groundtruth.max(),1,0)
    return frames, groundtruth


# In[27]:

##WRITE THE RIGHT PATH TO FOLDER 'DATA'!!!
data_list=load3Ddata('Data')
random.seed(1)
random.shuffle(data_list)
train_output=[]
test_output=[]
test_patients_number=[]
test_patients_spacing=[]
for j in range(len(data_list)):
    n=data_list[j][1]
    z=data_list[j][6]
    if j>=73:
        test_patients_number.append(data_list[j][0])
        test_patients_spacing.append(z)
    for h in range(n):
            EDslice=data_list[j][2][h]
            EDslicegt=data_list[j][3][h]
            ESslice=data_list[j][4][h]
            ESslicegt=data_list[j][5][h]
            if j<=72:
                train_output.append([data_list[j][0],h+1,EDslice,EDslicegt,ESslice,ESslicegt])
            else:
                test_output.append([data_list[j][0],h+1,EDslice,EDslicegt,ESslice,ESslicegt])

print(len(train_output))
print(len(test_output))


# In[28]:
##OPTIONAL: SAVE THE LISTS

with open('test_patients_number_list.pkl', 'wb') as f:
        pickle.dump(test_patients_number, f)
with open('test_patients_spacing_list.pkl', 'wb') as f:
        pickle.dump(test_patients_spacing, f)


# In[29]:


imgs_train, imgs_mask_train = load_list_as_dataset(train_output)
imgs_test, imgs_id_test = load_list_as_dataset(test_output)
print(imgs_train.shape)
print(imgs_mask_train.shape)
print(imgs_test.shape)
print(imgs_id_test.shape)


# In[ ]:


def train_fcn(a,b):

    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    model = fcn_model(a,b,weights=None)
    model_checkpoint = ModelCheckpoint('fcn_new_weights.h5', monitor=tversky_loss_2d, mode='min', save_best_only=True)

    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    #model.load_weights('fcn_new_weights.h5')
    hist=model.fit(imgs_train[:,:,:,np.newaxis], imgs_mask_train[:,:,:,np.newaxis],batch_size=5, epochs=10, verbose=1, shuffle=True,
              validation_split=0.3, callbacks=[model_checkpoint])


    print('-'*30)
    print('Save weights...')
    print('-'*30)
    model.save_weights('fcn_new_weights.h5')

    print('-'*30)
    print('Predicting masks on test data...')
    print('-'*30)
    imgs_mask_test = model.predict(imgs_test[:,:,:,np.newaxis], verbose=1)
    np.save('fcn_new_predictions.npy', imgs_mask_test)
    
    return hist

if __name__ == '__main__':
    hist_fcn_new=train_fcn((128,128,1),2)
    
    #Save the history to plot loss, accuracy or scores
    with open('history_fcn_new.pkl', 'wb') as f:
        pickle.dump(hist_fcn_new.history, f)


# In[243]:


plt.rcParams["figure.figsize"] = (5, 5)
x=hist_fcn_new.history
x_dice=x['dice_coef']
x_jaccard=x['jaccard_coef']

plt.plot(x_jaccard)
plt.plot(x_dice)
plt.title('model score')
plt.ylim(0., 1.)
plt.xlim(0,9)
plt.ylabel('score')
plt.xlabel('epoch')
plt.legend(['jaccard','dice'],loc='upper left')
plt.show()

