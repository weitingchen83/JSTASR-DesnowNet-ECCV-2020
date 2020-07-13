import numpy as np


#import Keras
import keras
from keras import backend as K
from keras.models import Model
from keras.models import load_model
from keras.layers import *

import cv2
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
class super_relu(Layer):
    def __init__(self, threshold0,threshold1,threshold2,threshold3,alpha, **kwargs):
        super(super_relu, self).__init__(**kwargs)
        self.threshold0 = K.cast_to_floatx(threshold0)
        self.threshold1 = K.cast_to_floatx(threshold1)
        self.threshold2 = K.cast_to_floatx(threshold2)
        self.threshold3 = K.cast_to_floatx(threshold3)
        self.alpha = K.cast_to_floatx(alpha)
        self.__name__ = 'super_relu'
    def call(self, inputs):
        parta=inputs*K.cast(K.greater(self.threshold0,inputs),'float32')
        partb=self.alpha*inputs*(K.cast(K.greater(self.threshold1,inputs),'float32')-K.cast(K.greater(self.threshold0,inputs),'float32'))
        partc=inputs*(120/(self.threshold2-self.threshold1))*(K.cast(K.greater(self.threshold2,inputs),'float32')-K.cast(K.greater(self.threshold1,inputs),'float32'))
        partd=(self.alpha*inputs+120)*(K.cast(K.greater(self.threshold3,inputs),'float32')-K.cast(K.greater(self.threshold2,inputs),'float32'))
        parte=inputs*K.cast(K.greater(inputs,self.threshold3),'float32')
    
        return parta+partb+partc+partd+parte

    def get_config(self):
        config = {'threshold0': float(self.threshold0),'threshold1': float(self.threshold1),'threshold2': float(self.threshold2),'threshold3': float(self.threshold3),'alpha': float(self.alpha)}
        base_config = super(super_relu, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape





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
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape

        
        
def __conv1_block(input,attentionParam, initializer):

    x = Multiply()([attentionParam, input])
    x = Conv2D(16, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(x)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x

def __conv2_block(input, k, kernel_size, strides_num, initializer, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 16 * k:
            init = Conv2D(16 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(init)
    else:
        if init._keras_shape[-1] != 16 * k:
            init = Conv2D(16 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(init)


    ## residual 3*3

    x = Conv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(16 * k, (3, 3), padding='same', strides = (strides_num, strides_num),dilation_rate=(3,3), kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)


    ## residual kernel_size * kernel_size

    
    x1 = Conv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    if dropout > 0.0:
        x1 = Dropout(dropout)(x1)

    x1 = Conv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x1)
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)
    



    ## residual kernel_size * kernel_size
    x2 = Conv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)
    
    x2 = Conv2D(16 * k, (5, 5), padding='same', strides = (strides_num, strides_num),dilation_rate=(2,2), kernel_initializer = initializer)(input)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x2)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    if dropout > 0.0:
        x2 = Dropout(dropout)(x2)


    m = add([init, x, x1, x2])

    return m


def __conv3_block(input, k, kernel_size, strides_num, initializer, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(init)
    else:
        if init._keras_shape[-1] != 32 * k:
            init = Conv2D(32 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(init)


    ## residual 3*3

    x = Conv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(32 * k, (3, 3), padding='same', strides = (strides_num, strides_num),dilation_rate=(2,2), kernel_initializer = initializer)(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x = Conv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    if dropout > 0.0:
        x = Dropout(dropout)(x)


    ## residual kernel_size * kernel_size

    x1 = Conv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    if dropout > 0.0:
        x1 = Dropout(dropout)(x1)

    x1 = Conv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x1)
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    ## residual kernel_size * kernel_size
    x2 = Conv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)


    x2 = Conv2D(32 * k, (5, 5), padding='same', strides = (strides_num, strides_num),dilation_rate=(3,3), kernel_initializer = initializer)(x2)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)
    
    x2 = Conv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x2)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    if dropout > 0.0:
        x2 = Dropout(dropout)(x2)
        
    m = add([init, x, x1, x2])

    return m


def ___conv4_block(input, k, kernel_size, strides_num, initializer, dropout=0.0):
    init = input

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Check if input number of filters is same as 16 * k, else create convolution2d for this input
    if K.image_data_format() == 'channels_first':
        if init._keras_shape[1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(init)
    else:
        if init._keras_shape[-1] != 64 * k:
            init = Conv2D(64 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(init)


    ## residual 3*3

    x = Conv2D(64 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    x = Conv2D(64 * k, (3, 3), padding='same', strides = (strides_num, strides_num),dilation_rate=(2,2), kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    
    x = Conv2D(64 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x)
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    if dropout > 0.0:
        x = Dropout(dropout)(x)


    ## residual kernel_size * kernel_size

    x1 = Conv2D(64 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    if dropout > 0.0:
        x1 = Dropout(dropout)(x1)

    x1 = Conv2D(64 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x1)
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    ## residual kernel_size * kernel_size
    x2 = Conv2D(64 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(input)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)


    x2 = Conv2D(64 * k, (5, 5), padding='same', strides = (strides_num, strides_num),dilation_rate=(2,2), kernel_initializer = initializer)(x2)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    x2 = Conv2D(64 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)(x2)
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    if dropout > 0.0:
        x2 = Dropout(dropout)(x2)

    m = add([init, x, x1, x2])

    return m

def __GCN(input, initializer, kernel_size_GCN):
    
    conv1_1 = Conv2D(64, (kernel_size_GCN, 1), padding='same', kernel_initializer = initializer)(input)
    conv1_2 = Conv2D(64, (1, kernel_size_GCN), padding='same', kernel_initializer = initializer)(input)
    conv2_1 = Conv2D(16, (1, kernel_size_GCN), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(16, (kernel_size_GCN, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    output = Concatenate()([conv2_1, conv2_2])

    return output

def __BR(input, initializer):
    
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)
    conv2 = Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    output = add([conv2, input])
    output = super_relu(threshold0=-120,threshold1=0,threshold2=60,threshold3=180,alpha=0.01)(output)
    return output

def __BR_1(input, initializer):
    
    conv1 = Conv2D(1, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)
    input = Conv2D(1, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)

    conv2 = Conv2D(1, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    output = add([conv2, input])
    output = Activation(bound_relu(maxvalue=120))(output)

    return output

def attention_path(input,initializer):
    x = Conv2D(16, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x_m = Activation('relu')(x)
    x1 = Conv2D(16, (2, 2), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x2 = Conv2D(16, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x3 = Conv2D(16, (5, 5), padding='same', activation='relu', kernel_initializer = initializer)(x_m)
    x_c = Concatenate()([x1,x2,x3])
    deconvATT_1 = Conv2DTranspose(32, (2, 2), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_1')(x_c)
    deconvATT_2 = Conv2DTranspose(32, (3, 3), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2')(x_c)
    mergeX = Concatenate()([deconvATT_1, deconvATT_2])
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(mergeX)
    conv2 = Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    BRatt = __BR(conv2, initializer)
    BRatt=Conv2D(1, (3, 3), padding='same', kernel_initializer = initializer)(BRatt)
    output = Activation(bound_relu(maxvalue=1.0))(BRatt)
    return output
    
    

    
    
    


def __create_patchMapNet(shape, img_input,
                                   width, dropout, kernel_size):
    


    kernel_size_deconv = 2
    strides_num = 1
    initializer = 'he_normal'
    
    attentionParam=attention_path(img_input, initializer)
    

    x = __conv1_block(img_input,attentionParam, initializer)

    

    x = MaxPool2D((2, 2), strides = (2,2))(x)

    res1 = __conv2_block(x, width, kernel_size, strides_num, initializer, dropout=0.25)
    
    #res1 = MaxPool2D((2, 2), strides = (2,2))(res1)
    #multiscale max pooling
    res1_1 = Conv2D(32, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(res1)
    res1_2 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(res1)
    res1_3 = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(res1)
    res1=Concatenate()([res1_1,res1_2,res1_3])
    res1 = Activation('relu')(res1)
    res1 = Dropout(0.4)(res1)
    

    res2 = __conv3_block(res1, width, kernel_size, strides_num, initializer, dropout=0.35)
    
    #res2 = MaxPool2D((2, 2), strides = (2,2))(res2)
    #multiscale max pooling
    res2_1 = Conv2D(32, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(res2)
    res2_2 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(res2)
    res2_3 = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(res2)
    res2=Concatenate()([res2_1,res2_2,res2_3])
    res2 = Activation('relu')(res2)
    res2 = Dropout(0.4)(res2)
    

    res3 = ___conv4_block(res2, width, kernel_size, strides_num, initializer, dropout=0.4)
    
    #res3 = MaxPool2D((2, 2), strides = (2,2))(res3)
    #multiscale max pooling
    res3_1 = Conv2D(32, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(res3)
    res3_2 = Conv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(res3)
    res3_3 = Conv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(res3)
    res3=Concatenate()([res3_1,res3_2,res3_3])
    res3 = Activation('relu')(res3)
    res3 = Dropout(0.4)(res3)


    deconv3 = Conv2DTranspose(64 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer,  name='deconv3')(res3)
    deconv3_1 = Conv2DTranspose(64 * width, (3, 3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer,  name='deconv3_1')(res3)

    merge3 = Concatenate()([deconv3, deconv3_1, res2])

    GCN3 = __GCN(res3, initializer, kernel_size_GCN = 7)
    BR3 = __BR(GCN3, initializer)
    Upsample3 = UpSampling2D(size=(2, 2), data_format=None)(BR3)
    
    
    GCN2 = __GCN(merge3, initializer, kernel_size_GCN = 15)
    BR2 = __BR(GCN2, initializer)
    Add2 = add([BR2, Upsample3])

    BR2_1 = __BR(Add2, initializer)

    Upsample2 = UpSampling2D(size=(2, 2), data_format=None)(BR2_1)


    deconv2 = Conv2DTranspose(32 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv2')(merge3)
    
    deconv2_1 = Conv2DTranspose(32 * width, (3, 3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv2_1')(merge3)
    
    merge2 = Concatenate()([deconv2,res1, deconv2_1])

    GCN1 = __GCN(merge2, initializer, kernel_size_GCN = 15)
    BR1 = __BR(GCN1, initializer)
    Add1 = add([BR1, Upsample2])
    BR1_1 = __BR(Add1, initializer)
    Upsample1 = UpSampling2D(size=(2, 2), data_format=None)(BR1_1)

    deconv4 = Conv2DTranspose(32 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv1')(merge2)

    deconv4_1 = Conv2DTranspose(32 * width, (3, 3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv1_1')(merge2)

    merge4 = Concatenate()([deconv4,x, deconv4_1])
    #ADD
    GCN4 = __GCN(merge4, initializer, kernel_size_GCN = 20)
    BR4 = __BR(GCN4, initializer)
    Add4 = add([BR4, Upsample1])

    BR4_1 = __BR(Add4, initializer)
    #Upsample4 = UpSampling2D(size=(2, 2), data_format=None)(BR4_1)
    deconv5 = Conv2DTranspose(8 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv5')(BR4_1)
    deconv5_1 = Conv2DTranspose(8 * width, (3,3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv5_1')(BR4_1)

    merge5 = Concatenate()([deconv5, deconv5_1])

    GCN5 = __GCN(merge5, initializer, kernel_size_GCN = 20)
    BR1_2 = __BR(GCN5, initializer)
    
    deconv6 = Conv2DTranspose(2 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv6')(BR1_2)
    deconv6_1 = Conv2DTranspose(2 * width, (3,3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv6_1')(BR1_2)

    merge6 = Concatenate()([deconv6, deconv6_1])
    
    
    BR1_3 = __BR_1(merge6, initializer)

    return BR1_3
    
def build_patchMapNet_10(shape, lr=0.0001):
    img_input = Input(shape=shape)
    x= __create_patchMapNet(shape, img_input=img_input, width=10,dropout=0.5, kernel_size=1) 
    inputs = img_input
    model = Model(inputs, [x], name='patchMapNet')
    return model 
