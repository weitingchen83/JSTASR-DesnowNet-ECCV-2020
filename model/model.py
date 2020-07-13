import matplotlib.pyplot as plt
import numpy as np
import sys
import os

import tensorflow as tf
import keras
from keras import backend as K
from keras.models import Sequential,Model
from keras.layers import *
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects

import scipy.misc
import scipy.io
import cv2

class addLayer(Layer):
    def __init__(self, add_value, **kwargs):
        super(addLayer, self).__init__(**kwargs)
        self.add_value = K.cast_to_floatx(add_value)
        self.__name__ = 'addLayer'

    def call(self, inputs):
        return inputs+self.add_value

    def get_config(self):
        config = {'add_value': float(self.add_value)}
        base_config = super(addLayer, self).get_config()
        return dict(list(base_config.items())+ list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
  
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


#recover pre
class RecoverLayer(Layer):
    def __init__(self, **kwargs):
        super(RecoverLayer, self).__init__(**kwargs)
        self.__name__ = 'RecoverLayer'

    def call(self, PMandHazeIMGandA):
        
        patchMap=PMandHazeIMGandA[0]
        hazeImg=PMandHazeIMGandA[1]
        A=PMandHazeIMGandA[2]
        x=tf.cast(patchMap,tf.int32)
        s=K.int_shape(patchMap)
        mapList=[]
        for i in range(1,121):
            if i == 1:
                mapList.append(K.greater(1.3,patchMap))
            else:
                mapList.append(K.equal(x,i))
        pmbox=K.concatenate(mapList,axis=3)
        pmbox=tf.cast(pmbox,tf.float32)
        patchConvList=[]
        patchMinList=[]
        
        """
        hazeImg=-hazeImg
        
        for i in range(1,121):
            recoveri=K.pool2d(hazeImg, pool_size=(i,i), strides=(1,1),
                          padding='same', data_format=None,
                          pool_mode='max')
            recoveri=-recoveri
            recoveri=K.min(recoveri,axis=3,keepdims=True)
            patchMinList.append(recoveri)
        """
        hazeImgM = K.min(hazeImg,axis=3,keepdims=True)
        hazeImgM=-hazeImgM
        for i in range(1,121):
            if i <= 5:
                recoveri=K.pool2d(hazeImgM, pool_size=(i,i), strides=(1,1),
                          padding='same', data_format=None,
                          pool_mode='max')
                #recoveri=-recoveri
                #recoveri=K.min(recoveri,axis=3,keepdims=True)
                patchMinList.append(recoveri)
            else:
                #print(len(patchMinList),'len patchMinList')
                #recoveri=-patchMinList[i-2]
                recoveri=K.pool2d(patchMinList[i-3], pool_size=(3,3), strides=(1,1),
                          padding='same', data_format=None,
                          pool_mode='max')
                #recoveri=-recoveri
                #recoveri=K.min(recoveri,axis=3,keepdims=True)
                patchMinList.append(recoveri)
                
        #for i in range(120):
        #    patchMinList[i]=K.min(-patchMinList[i],axis=3,keepdims=True)
        
        
        #hazeImg=-hazeImg
        rcbox=K.concatenate(patchMinList,axis=3)
        #print(K.int_shape(rcbox),'int shape')
        rcbox = -rcbox
        
        A2=K.expand_dims(A,2)
        A2=K.expand_dims(A2,3)
        A2=K.tile(A2,[1,480,640,120])
        t=rcbox/A2
        w=31/32
        t=w*t
        t=1-t
        pm_rc0=pmbox*t
        pm_rc=K.sum(pm_rc0,axis=3,keepdims=True)
        pm_rc=K.clip(pm_rc,0.0,1.0)
        return [pm_rc]

    def get_config(self):
        base_config = super(RecoverLayer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[0]]
#Recover Pre End



#Recover Post
class RecoverPostLayer(Layer):
    def __init__(self, **kwargs):
        super(RecoverPostLayer, self).__init__(**kwargs)
        self.__name__ = 'RecoverPostLayer'

    def call(self, TandHazeIMGandA):
        
        T=TandHazeIMGandA[0]
        hazeImg=TandHazeIMGandA[1]
        A=TandHazeIMGandA[2]
        pm_rc=K.clip(T,0.1,1.0)
        A=K.expand_dims(A,2)
        A=K.expand_dims(A,3)
        A=K.tile(A,[1,480,640,3])
        tmp=(hazeImg-A)
        tmp=tmp/pm_rc
        recoverImg=tmp+A
        return [recoverImg,pm_rc]

    def get_config(self):
        base_config = super(RecoverPostLayer, self).get_config()
        return dict(list(base_config.items()))

    def compute_output_shape(self, input_shape):
        return [input_shape[1],input_shape[0]]
#RECOVER Post END

def build_combine_model():
 #BUILD COMBINE MODEL

    import model.model_PM as model_PM
    model = model_PM.build_patchMapNet_10((480,640,3))
    modelA = load_model('./modelParam/A.h5')
    import model.model_TR as model_TR
    modelTR = model_TR.build_TR((480,640,4))
    
    inp=Input(shape=(480,640,3))
    A=modelA(inp)
    pm=model(inp)
    pmrc=RecoverLayer()([pm,inp,A])
    TRinp=Concatenate(axis=3)([pmrc,inp])
    pmrcRefine=modelTR(TRinp)
    recov,postT=RecoverPostLayer()([pmrcRefine,inp,A])
    modelRecoverCombine=Model(inputs=[inp],outputs=[recov])
    modelRecoverCombine.load_weights('./modelParam/finalModel.h5',by_name=False)
    #recov=recov[:,:,:,::-1]
    recov=Conv2D(3, (1), padding='same', kernel_initializer = keras.initializers.Constant(value=[[0,0,1],[0,1,0],[1,0,0]]), use_bias=False)(recov)
    recov=Activation(bound_relu(maxvalue=1.0))(recov)

    #Build Model
    
    import model.model_desnow as ms
    modelSnow=ms.build_CombineModel((480,640,3),0.1)
    modelSnow.load_weights('./modelParam/snowmodel.h5',by_name=False)
    
    x=modelSnow([recov])
    modelFinal=Model(inputs=[inp],outputs=x)
    #modelFinal.summary()



    return modelFinal
