"""
Fully Convolutional Neural Network

Team Challenge (TU/e & UU)
Team 2
"""

import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras import backend as K
import keras
from keras import optimizers
from keras.layers import Lambda, ZeroPadding2D

K.set_image_data_format('channels_last')  # TF dimension ordering in this code

# -----------------------------------------------------------------------------

# Function which performs the spatial mean-variance normalization per channel
def mvn(tensor):
    epsilon = 1e-6
    mean = K.mean(tensor, axis=(1,2), keepdims=True)
    std = K.std(tensor, axis=(1,2), keepdims=True)
    mvn = (tensor - mean) / (std + epsilon)

    return mvn

# Function which computes the average dice coefficient per batch
def dice_coef(y_true, y_pred, smooth=0.0):
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    summation = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes)
    
    return K.mean((2.0 * intersection + smooth) / (summation + smooth), axis=0)

# Function which computes the average dice coefficient loss per batch
def dice_coef_loss(y_true, y_pred):
    return 1.0 - dice_coef(y_true, y_pred, smooth=10.0)

# Function which computes the average jaccard coefficient per batch
def jaccard_coef(y_true, y_pred, smooth=0.0):
    axes = (1,2,3)
    intersection = K.sum(y_true * y_pred, axis=axes)
    union = K.sum(y_true, axis=axes) + K.sum(y_pred, axis=axes) - intersection
    return K.mean( (intersection + smooth) / (union + smooth), axis=0)

# Function which computes the tversky coefficient loss per batch
def tversky_loss(Y_gt, Y_pred):
    smooth = 1e-5
    alpha = 0.5
    beta = 0.5
    ones = tf.ones(tf.shape(Y_gt))
    p0 = Y_pred
    p1 = ones - Y_pred
    g0 = Y_gt
    g1 = ones - Y_gt
    num = tf.reduce_sum(p0 * g0, axis=[1, 2])
    den = num + alpha * tf.reduce_sum(p0 * g1, axis=[1, 2]) + \
          beta * tf.reduce_sum(p1 * g0, axis=[1, 2]) + smooth
    tversky = tf.reduce_sum(num / den, axis=1)
    loss = tf.reduce_mean(1 - tversky)
    return loss

# -----------------------------------------------------------------------------

# Function to define the FCN network
def fcn_model(input_shape, num_classes, weights=None):
    ''' FCN architecture based on a study of Long et al., 2015
    https://arxiv.org/abs/1411.4038
    '''
    if num_classes == 2:
        num_classes = 1
        loss = tversky_loss
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

