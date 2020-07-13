import numpy as np
import scipy
from scipy import stats
import pandas as pd
import argparse
import time
import math
from copy import deepcopy
import sys
import os
import gc
import pickle
from sklearn.model_selection import train_test_split

#import Keras
import keras
from keras import backend as K
from keras.models import Sequential,Model
import cv2
from keras.layers import *

class bound_relu(Layer):
    def __init__(self, maxvalue, **kwargs):
        super(bound_relu, self).__init__(**kwargs)
        self.maxvalue = K.cast_to_floatx(maxvalue)
        self.__name__ = 'bound_relu'

    def call(self, inputs):
        return keras.activations.relu(inputs, max_value=self.maxvalue)

    def get_config(self):
        config = {'maxvalue': float(self.maxvalue)}
        base_config = super(bound_relu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class binary_mul_layer(Layer):
    def __init__(self, mulvalue,threshold=0.05, **kwargs):
        super(binary_mul_layer, self).__init__(**kwargs)
        self.mulvalue = K.cast_to_floatx(mulvalue)
        self.threshold = K.cast_to_floatx(threshold)
        self.__name__ = 'binary_mul_layer'

    def call(self, inputs):
        SnowInput=inputs
        binarySnow=K.cast(K.greater(SnowInput,self.threshold),'float32')
        mulBinarySnow=binarySnow*self.mulvalue
        return mulBinarySnow
    
    def get_config(self):
        config = {'mulvalue': float(self.mulvalue),'threshold': float(self.threshold)}
        base_config = super(binary_mul_layer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


class comp_layer(Layer):
    def __init__(self, **kwargs):
        super(comp_layer, self).__init__(**kwargs)
        self.__name__ = 'comp_layer'

    def call(self, input):
        
        
        return 1-input
    
    def get_config(self):
        base_config = super(comp_layer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape



# And the Mask and Final Mask
class AndLayer(Layer):
    def __init__(self, andvalue, **kwargs):
        super(AndLayer, self).__init__(**kwargs)
        self.andvalue = K.cast_to_floatx(andvalue)
        self.__name__ = 'AndLayer'

        
    def call(self, inputs):
        totalmaskInput=inputs[0]
        out=K.cast(K.equal(self.andvalue,totalmaskInput),'float32')
        return out
        
    def get_config(self):
        base_config = super(AndLayer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape


#Binary Dig
class DigLayer(Layer):
    def __init__(self,threshold=0.05, **kwargs):
        super(DigLayer, self).__init__(**kwargs)
        self.threshold = K.cast_to_floatx(threshold)
        self.__name__ = 'DigLayer'

        
    def call(self, inputs):
        imgInput=inputs[0]
        maskInput=inputs[1]
        binarySnow=K.cast(K.greater(maskInput,self.threshold),'float32')
        partHole=1-binarySnow
        return imgInput*partHole
        
    def get_config(self):
        config = {'threshold': float(self.threshold)}
        base_config = super(DigLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]

class BinaryMaskMulLayer(Layer):
    def __init__(self,threshold=0.05, **kwargs):
        super(BinaryMaskMulLayer, self).__init__(**kwargs)
        self.threshold = K.cast_to_floatx(threshold)
        self.__name__ = 'BinaryMaskMulLayer'

        
    def call(self, inputs):
        imgInput=inputs[0]
        maskInput=inputs[1]
        binarySnow=K.cast(K.greater(maskInput,self.threshold),'float32')
        return imgInput*binarySnow
        
    def get_config(self):
        config = {'threshold': float(self.threshold)}
        base_config = super(BinaryMaskMulLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape[0]




def attention_path_S(input,initializer,nameAdjust):
    
    x = Conv2D(6, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x2 = Conv2D(6, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x2 = Conv2D(6, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)

    x2 = Conv2D(6, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x = Dropout(0.25)(x)
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x_m = Activation('relu')(x)
    
    x1 = Conv2D(8, (2, 2), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x2 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x3 = Conv2D(8, (5, 5), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    
    x_c = Concatenate()([x1,x2,x3])
    x_c = Dropout(0.3)(x_c)
    
    deconvATT_1 = Conv2DTranspose(8, (2, 2), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_1'+nameAdjust)(x_c)
    deconvATT_2 = Conv2DTranspose(8, (3, 3), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2'+nameAdjust)(x_c)
    deconvATT_3 = Conv2DTranspose(8, (5, 5), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_3'+nameAdjust)(x_c)
    
    mergeX = Concatenate()([deconvATT_1, deconvATT_2, deconvATT_3])
    
    conv1_1 = Conv2D(16, (7, 1), padding='same', kernel_initializer = initializer)(mergeX)
    conv1_2 = Conv2D(16, (1, 7), padding='same', kernel_initializer = initializer)(mergeX)
    conv2_1 = Conv2D(4, (1, 7), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(4, (7, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    GCN = Concatenate()([conv2_1, conv2_2])
    
    GCN = BatchNormalization(axis=channel_axis)(GCN)
    GCN = Dropout(0.4)(GCN)
    GCN = Activation('relu')(GCN)
    
    
    conv1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(GCN)
    conv2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    #output = add([conv2, mergeX])
    
    #BRatt = __BR(conv2, initializer)
    c1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(conv2)
    c2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(c1)
    BRatt = add([c2, conv2])
    
    BRatt=Conv2D(1, (3, 3), padding='same', kernel_initializer = initializer)(BRatt)
    
    output = Activation(bound_relu(maxvalue=1.0))(BRatt)
    #output = addLayer(add_value=0.5)(output) 
    
    return output
    

def attention_path_M(input,initializer,nameAdjust):

    x = Conv2D(6, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x2 = Conv2D(6, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x3 = Conv2D(6, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)

    x = Concatenate()([x,x2,x3])
    x = Dropout(0.25)(x)
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x_m = Activation('relu')(x)
    
    x1 = Conv2D(8, (2, 2), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x2 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x3 = Conv2D(8, (5, 5), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    
    x_c = Concatenate()([x1,x2,x3])
    x_c = Dropout(0.3)(x_c)
    
    deconvATT_1 = Conv2DTranspose(8, (2, 2), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_1'+nameAdjust)(x_c)
    deconvATT_2 = Conv2DTranspose(8, (3, 3), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2'+nameAdjust)(x_c)
    deconvATT_3 = Conv2DTranspose(8, (5, 5), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_3'+nameAdjust)(x_c)
    
    mergeX = Concatenate()([deconvATT_1, deconvATT_2, deconvATT_3])
    
    
    conv1_1 = Conv2D(16, (12, 1), padding='same', kernel_initializer = initializer)(mergeX)
    conv1_2 = Conv2D(16, (1, 12), padding='same', kernel_initializer = initializer)(mergeX)
    conv2_1 = Conv2D(4, (1, 12), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(4, (12, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    GCN = Concatenate()([conv2_1, conv2_2])
    
    GCN = BatchNormalization(axis=channel_axis)(GCN)
    GCN = Dropout(0.4)(GCN)
    GCN = Activation('relu')(GCN)
    
    
    conv1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(GCN)
    conv2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    #output = add([conv2, mergeX])
    
    #BRatt = __BR(conv2, initializer)
    c1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(conv2)
    c2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(c1)
    BRatt = add([c2, conv2])
    
    BRatt=Conv2D(1, (3, 3), padding='same', kernel_initializer = initializer)(BRatt)
    
    output = Activation(bound_relu(maxvalue=1.0))(BRatt)
    #output = addLayer(add_value=0.5)(output) 
    
    return output
    
    
def attention_path_L(input,initializer,nameAdjust):
    x = Conv2D(6, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x2 = Conv2D(6, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x3 = Conv2D(6, (7, 7), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)

    x = Concatenate()([x,x2,x3])
    x = Dropout(0.25)(x)
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x_m = Activation('relu')(x)
    
    #x_m = MaxPool2D((2, 2), strides = (2,2))(x)
    
    x1 = Conv2D(6, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x2 = Conv2D(6, (7, 7), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x3 = Conv2D(6, (15, 15), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    
    x_c = Concatenate()([x1,x2,x3])
    x_c = Dropout(0.3)(x_c)
    
    deconvATT_1 = Conv2DTranspose(8, (3, 3), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_1'+nameAdjust)(x_c)
    deconvATT_2 = Conv2DTranspose(8, (5, 5), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2'+nameAdjust)(x_c)
    deconvATT_3 = Conv2DTranspose(8, (14, 14), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_3'+nameAdjust)(x_c)
    
    mergeX = Concatenate()([deconvATT_1, deconvATT_2, deconvATT_3])
    
    
    
    conv1_1 = Conv2D(16, (15, 1), padding='same', kernel_initializer = initializer)(mergeX)
    conv1_2 = Conv2D(16, (1, 15), padding='same', kernel_initializer = initializer)(mergeX)
    conv2_1 = Conv2D(4, (1, 15), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(4, (15, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    GCN = Concatenate()([conv2_1, conv2_2])
    
    GCN = BatchNormalization(axis=channel_axis)(GCN)
    GCN = Dropout(0.4)(GCN)
    GCN = Activation('relu')(GCN)
    
    
    conv1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(GCN)
    conv2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    #output = add([conv2, mergeX])
    
    #BRatt = __BR(conv2, initializer)
    c1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(conv2)
    c2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(c1)
    BRatt = add([c2, conv2])
    
    BRatt=Conv2D(1, (3, 3), padding='same', kernel_initializer = initializer)(BRatt)
    
    output = Activation(bound_relu(maxvalue=1.0))(BRatt)
    #output = addLayer(add_value=0.5)(output) 
    
    return output
 
 

def SnowAttentionModel(img_input):
    
    attB=attention_path_L(img_input, 'he_normal','BSnow')
    attM=attention_path_M(img_input, 'he_normal','MSnow')
    attS=attention_path_S(img_input, 'he_normal','SSnow')
    
    
    return attB,attM,attS



def build_SnowAttentionModel(shape):
    print('Build SnowAttentionModel')
    img_input = Input(shape=shape)
    attB,attM,attS=SnowAttentionModel(img_input)
    #inputs = [BSnow_input,MSnow_input,SSnow_input]
    inputs = img_input
    # Create model.
    model = Model(inputs, [attB,attM,attS], name='SnowAttNet')
    return model 