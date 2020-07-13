import numpy as np
import keras
from keras import backend as K
from keras.models import load_model
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


def __BR(input, initializer):
    
    conv1 = Conv2D(32, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)
    conv2 = Conv2D(32, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    output = add([conv2, input])
    return output

def __GCN(input, initializer, kernel_size_GCN):
    conv1_1 = Conv2D(32, (kernel_size_GCN, 1), padding='same', kernel_initializer = initializer)(input)
    conv1_2 = Conv2D(32, (1, kernel_size_GCN), padding='same', kernel_initializer = initializer)(input)
    conv2_1 = Conv2D(16, (1, kernel_size_GCN), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(16, (kernel_size_GCN, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    output = Concatenate()([conv2_1, conv2_2])

    return output




def attention_path(input,initializer):
    x = Conv2D(16, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x2 = Conv2D(16, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)
    x3 = Conv2D(16, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)(input)

    x = Concatenate()([x,x2,x3])

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x_m = Activation('relu')(x)
    x1 = Conv2D(16, (2, 2), padding='same', strides=(2, 2), activation='relu', kernel_initializer = initializer)(x_m)
    x2 = Conv2D(16, (3, 3), padding='same', strides=(2, 2), activation='relu', kernel_initializer = initializer)(x_m)
    x3 = Conv2D(16, (5, 5), padding='same', strides=(2, 2), activation='relu', kernel_initializer = initializer)(x_m)
    x_c = Concatenate()([x1,x2,x3])
    res1 = __conv2_block(x_c, 8, 1, 1, initializer, dropout=0.35)
    
    deconvATT_1 = Conv2DTranspose(32, (2, 2), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_1')(res1)
    deconvATT_2 = Conv2DTranspose(32, (3, 3), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2')(res1)
    
    mergeX = Concatenate()([deconvATT_1, deconvATT_2,x_m])
    
    
    GCN1 = __GCN(mergeX, initializer, kernel_size_GCN = 15)
    BR1 = __BR(GCN1, initializer)
    
    
    GCN2 = __GCN(res1, initializer, kernel_size_GCN = 15)
    BR3 = __BR(GCN2, initializer)
    Upsample3 = UpSampling2D(size=(2, 2), data_format=None)(BR3)
    
    
    con2 = Concatenate()([BR1, Upsample3])
    
    GCN3 = __GCN(con2, initializer, kernel_size_GCN = 30)
    BR2_1 = __BR(GCN3, initializer)

    Upsample2 = UpSampling2D(size=(2, 2), data_format=None)(BR2_1)

    
    deconv2 = Conv2DTranspose(32, (2, 2), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2_0')(mergeX)

    deconv2_1 = Conv2DTranspose(32, (3, 3), padding='same', strides = 2, activation='relu', kernel_initializer = initializer, name='deconvATT_2_1')(mergeX)

    merge2 = Concatenate()([deconv2, deconv2_1,Upsample2])

    GCNatt = __GCN(merge2, initializer, kernel_size_GCN = 30)
    BRatt = __BR(GCNatt, initializer)
    
    
    convlast1 = Conv2D(1, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(BRatt)
    BRatt = Conv2D(1, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(BRatt)

    convlast2 = Conv2D(1, (3, 3), padding='same', kernel_initializer = initializer)(convlast1)
    output = add([convlast2, BRatt])
    
    output = Activation(bound_relu(maxvalue=1.0))(output)
    return output


def build_TR(shape, lr=0.0001):
    print('Build TR')
    img_input = Input(shape=shape)
    
    initializer = 'he_normal'
    x=attention_path(img_input,initializer=initializer)
    inputs = img_input
    # Create model.
    model = Model(inputs, [x], name='TRNet')
    return model 

