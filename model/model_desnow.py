import numpy as np
import keras
from keras import backend as K
from keras.models import Model
from keras.models import load_model
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

#Partial Conv 2D
class PConv2D(Conv2D):
    def __init__(self, *args, n_channels=3, mono=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.input_spec = [InputSpec(ndim=4), InputSpec(ndim=4)]

    def build(self, input_shape):    
        if self.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1
            
        if input_shape[0][channel_axis] is None:
            raise ValueError('The channel dimension of the inputs should be defined. Found `None`.')
            
        self.input_dim = input_shape[0][channel_axis]
        
        # Image kernel
        kernel_shape = self.kernel_size + (self.input_dim, self.filters)
        self.kernel = self.add_weight(shape=kernel_shape,
                                      initializer=self.kernel_initializer,
                                      name='img_kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        # Mask kernel
        self.kernel_mask = K.ones(shape=self.kernel_size + (self.input_dim, self.filters))

        # Calculate padding size to achieve zero-padding
        self.pconv_padding = (
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
            (int((self.kernel_size[0]-1)/2), int((self.kernel_size[0]-1)/2)), 
        )

        # Window size - used for normalization
        self.window_size = self.kernel_size[0] * self.kernel_size[1]
        
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.filters,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.built = True

    def call(self, inputs, mask=None):
        # Both image and mask must be supplied
        if type(inputs) is not list or len(inputs) != 2:
            raise Exception('PartialConvolution2D must be called on a list of two tensors [img, mask]. Instead got: ' + str(inputs))

        # Padding done explicitly so that padding becomes part of the masked partial convolution
        images = K.spatial_2d_padding(inputs[0], self.pconv_padding, self.data_format)
        masks = K.spatial_2d_padding(inputs[1], self.pconv_padding, self.data_format)

        # Apply convolutions to mask
        mask_output = K.conv2d(
            masks, self.kernel_mask, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )

        # Apply convolutions to image
        img_output = K.conv2d(
            (images*masks), self.kernel, 
            strides=self.strides,
            padding='valid',
            data_format=self.data_format,
            dilation_rate=self.dilation_rate
        )        

        # Calculate the mask ratio on each pixel in the output mask
        mask_ratio = self.window_size / (mask_output + 1e-8)

        # Clip output to be between 0 and 1
        mask_output = K.clip(mask_output, 0, 1)

        # Remove ratio values where there are holes
        mask_ratio = mask_ratio * mask_output

        # Normalize iamge output
        img_output = img_output * mask_ratio

        # Apply bias only to the image (if chosen to do so)
        if self.use_bias:
            img_output = K.bias_add(
                img_output,
                self.bias,
                data_format=self.data_format)
        
        # Apply activations on the image
        if self.activation is not None:
            img_output = self.activation(img_output)
        return [img_output, mask_output]
    
    def compute_output_shape(self, input_shape):
        if self.data_format == 'channels_last':
            space = input_shape[0][1:-1]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0][0],) + tuple(new_space) + (self.filters,)
            return [new_shape, new_shape]
        if self.data_format == 'channels_first':
            space = input_shape[2:]
            new_space = []
            for i in range(len(space)):
                new_dim = conv_utils.conv_output_length(
                    space[i],
                    self.kernel_size[i],
                    padding='same',
                    stride=self.strides[i],
                    dilation=self.dilation_rate[i])
                new_space.append(new_dim)
            new_shape = (input_shape[0], self.filters) + tuple(new_space)
            return [new_shape, new_shape]




  
def __conv1_block(input,mask_input, initializer):

    x,mask1 = PConv2D(8, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)([input,mask_input])
    x2,mask2 = PConv2D(8, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)([input,mask_input])
    x3,mask3 = PConv2D(8, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)([input,mask_input])

    x = Concatenate()([x,x2,x3])
    mask = Concatenate()([mask1,mask2,mask3])
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    return x,mask

def __conv2_block(input,mask_input, k, kernel_size, strides_num, initializer, dropout=0.0):

    init = input
    init,mask0 = PConv2D(16 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([init,mask_input])
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    ## residual 3*3
    x,mask1 = PConv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x,mask1 = PConv2D(16 * k, (3, 3), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x,mask1])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x,mask1 = PConv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x,mask1])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    if dropout > 0.0:
        x = Dropout(dropout)(x)


    ## residual kernel_size * kernel_size
    x1,mask2 = PConv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    if dropout > 0.0:
        x1 = Dropout(dropout)(x1)

    x1,mask2 = PConv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x1,mask2])
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)
    



    ## residual kernel_size * kernel_size
    x2,mask3 = PConv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)
    
    x2,mask3 = PConv2D(16 * k, (5, 5), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x2,mask3])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    x2,mask3 = PConv2D(16 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x2,mask3])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    if dropout > 0.0:
        x2 = Dropout(dropout)(x2)
    m = add([init,x, x1, x2])

    return m,mask3


def __conv3_block(input,mask_input, k, kernel_size, strides_num, initializer, dropout=0.0):
    

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    init = input
    init,mask0 = PConv2D(32 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([init,mask_input])
    ## residual 3*3

    x,mask1 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x,mask1 = PConv2D(32 * k, (3, 3), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x,mask1])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)

    x,mask1 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x,mask1])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    if dropout > 0.0:
        x = Dropout(dropout)(x)


    ## residual kernel_size * kernel_size

    x1,mask2 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    if dropout > 0.0:
        x1 = Dropout(dropout)(x1)

    x1,mask2 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x1,mask2])
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    ## residual kernel_size * kernel_size
    x2,mask3 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)


    x2,mask3 = PConv2D(32 * k, (5, 5), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x2,mask3])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)
    
    x2,mask3 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x2,mask3])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    if dropout > 0.0:
        x2 = Dropout(dropout)(x2)
        
    m = add([init,x, x1, x2])

    return m,mask3


def ___conv4_block(input,mask_input, k, kernel_size, strides_num, initializer, dropout=0.0):
    
    
    init = input
    init,mask0 = PConv2D(32 * k, (1, 1), activation='linear', padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([init,mask_input])
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1


    ## residual 3*3

    x,mask1 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    x,mask1 = PConv2D(32 * k, (3, 3), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x,mask1])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)
    
    x,mask1 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x,mask1])
    x = BatchNormalization(axis=channel_axis)(x)
    x = Activation('relu')(x)


    if dropout > 0.0:
        x = Dropout(dropout)(x)


    ## residual kernel_size * kernel_size

    x1,mask2 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    if dropout > 0.0:
        x1 = Dropout(dropout)(x1)

    x1,mask2 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x1,mask2])
    x1 = BatchNormalization(axis=channel_axis)(x1)
    x1 = Activation('relu')(x1)

    ## residual kernel_size * kernel_size
    x2,mask3 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([input,mask_input])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)


    x2,mask3 = PConv2D(32 * k, (5, 5), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x2,mask3])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    x2,mask3 = PConv2D(32 * k, (1, 1), padding='same', strides = (strides_num, strides_num), kernel_initializer = initializer)([x2,mask3])
    x2 = BatchNormalization(axis=channel_axis)(x2)
    x2 = Activation('relu')(x2)

    if dropout > 0.0:
        x2 = Dropout(dropout)(x2)
    m = add([init,x, x1, x2])

    return m,mask3


def __GCN(input, initializer, kernel_size_GCN):
    conv1_1 = Conv2D(16, (kernel_size_GCN, 1), padding='same', kernel_initializer = initializer)(input)
    conv1_2 = Conv2D(16, (1, kernel_size_GCN), padding='same', kernel_initializer = initializer)(input)
    conv2_1 = Conv2D(4, (1, kernel_size_GCN), padding='same', kernel_initializer = initializer)(conv1_1)
    conv2_2 = Conv2D(4, (kernel_size_GCN, 1), padding='same', kernel_initializer = initializer)(conv1_2)
    output = Concatenate()([conv2_1, conv2_2])

    return output




def __BR(input, initializer):
    
    conv1 = Conv2D(8, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)
    conv2 = Conv2D(8, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    output = add([conv2, input])
    

    return output

def __BR_1(input, initializer):
    
    conv1 = Conv2D(3, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)
    input = Conv2D(3, (3, 3), padding='same', activation='relu', kernel_initializer = initializer)(input)

    conv2 = Conv2D(3, (3, 3), padding='same', kernel_initializer = initializer)(conv1)
    output = add([conv2, input])
    

    return output



def Deep_Desnow_Model_B(img_input,mask_input,mask_inputCombine,mask_inputCombine_different_thres,namePostFix):
    
    

    #N = (depth - 4) // 6

    kernel_size_deconv = 2
    strides_num = 1
    initializer = 'he_normal'
    width=7
    kernel_size=1
    comp_mask_input=comp_layer()(mask_inputCombine_different_thres)
    
    x,mask = __conv1_block(img_input,comp_mask_input, initializer)

    #multiscale max pooling
    x_1,mask1 = PConv2D(16, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)([x,mask])
    x_2,mask2= PConv2D(16, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)([x,mask])
    x_3,mask3 = PConv2D(16, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)([x,mask])
    x_4,mask4 = PConv2D(16, (7, 7), padding='same', strides=(2, 2), kernel_initializer = initializer)([x,mask])
    x=Concatenate()([x_1,x_2,x_3,x_4])
    x_mask=Concatenate()([mask1,mask2,mask3,mask4])
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)


    res1,res1_mask = __conv2_block(x,x_mask, width, kernel_size, strides_num, initializer, dropout=0.25)
    
    #multiscale max pooling
    res1_1,res1_mask1 = PConv2D(24, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)([res1,res1_mask])
    res1_2,res1_mask2 = PConv2D(24, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)([res1,res1_mask])
    res1_3,res1_mask3 = PConv2D(24, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)([res1,res1_mask])
    res1_4,res1_mask4 = PConv2D(24, (7, 7), padding='same', strides=(2, 2), kernel_initializer = initializer)([res1,res1_mask])
    res1=Concatenate()([res1_1,res1_2,res1_3,res1_4])
    res1_mask=Concatenate()([res1_mask1,res1_mask2,res1_mask3,res1_mask4])
    res1 = Activation('relu')(res1)
    res1 = Dropout(0.3)(res1)
    

    res2,res2_mask = __conv3_block(res1,res1_mask, width, kernel_size, strides_num, initializer, dropout=0.35)
    
    #multiscale max pooling
    res2_1,res2_mask1 = PConv2D(28, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)([res2,res2_mask])
    res2_2,res2_mask2 = PConv2D(28, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)([res2,res2_mask])
    res2_3,res2_mask3 = PConv2D(28, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)([res2,res2_mask])
    res2_4,res2_mask4 = PConv2D(28, (7, 7), padding='same', strides=(2, 2), kernel_initializer = initializer)([res2,res2_mask])
    res2=Concatenate()([res2_1,res2_2,res2_3,res2_4])
    res2_mask=Concatenate()([res2_mask1,res2_mask2,res2_mask3,res2_mask4])
    res2 = Activation('relu')(res2)
    res2 = Dropout(0.4)(res2)
    
    res3,res3_mask = ___conv4_block(res2,res2_mask, width, kernel_size, strides_num, initializer, dropout=0.4)
    
    #multiscale max pooling
    res3_1,res3_mask1 = PConv2D(32, (2, 2), padding='same', strides=(2, 2), kernel_initializer = initializer)([res3,res3_mask])
    res3_2,res3_mask2 = PConv2D(32, (3, 3), padding='same', strides=(2, 2), kernel_initializer = initializer)([res3,res3_mask])
    res3_3,res3_mask3 = PConv2D(32, (5, 5), padding='same', strides=(2, 2), kernel_initializer = initializer)([res3,res3_mask])
    res3_4,res3_mask4 = PConv2D(32, (7, 7), padding='same', strides=(2, 2), kernel_initializer = initializer)([res3,res3_mask])
    res3=Concatenate()([res3_1,res3_2,res3_3,res3_4])
    res3_mask=Concatenate()([res3_mask1,res3_mask2,res3_mask3,res3_mask4])
    res3 = Activation('relu')(res3)
    res3 = Dropout(0.4)(res3)

    deconv3 = Conv2DTranspose(32 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer,  name='deconv3'+namePostFix)(res3)
    deconv3_1 = Conv2DTranspose(32 * width, (3, 3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer,  name='deconv3_1'+namePostFix)(res3)
    deconv3_2 = Conv2DTranspose(32 * width, (5, 5), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer,  name='deconv3_2'+namePostFix)(res3)

    res2_combineMask = Multiply()([res2,res2_mask])
    merge3 = Concatenate()([deconv3, deconv3_1, res2_combineMask, deconv3_2])

    GCN3 = __GCN(res3, initializer, kernel_size_GCN = 7)
    BR3 = __BR(GCN3, initializer)
    Upsample3 = UpSampling2D(size=(2, 2), data_format=None)(BR3)
    
    
    GCN2 = __GCN(merge3, initializer, kernel_size_GCN = 15)
    BR2 = __BR(GCN2, initializer)
    Add2 = add([BR2, Upsample3])

    BR2_1 = __BR(Add2, initializer)

    Upsample2 = UpSampling2D(size=(2, 2), data_format=None)(BR2_1)


    deconv2 = Conv2DTranspose(24 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv2'+namePostFix)(merge3)
    
    deconv2_1 = Conv2DTranspose(24 * width, (3, 3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv2_1'+namePostFix)(merge3)
    deconv2_2 = Conv2DTranspose(24 * width, (5, 5), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv2_2'+namePostFix)(merge3)
    
    res1_combineMask = Multiply()([res1,res1_mask])
    merge2 = Concatenate()([deconv2,res1_combineMask, deconv2_1,deconv2_2])

    GCN1 = __GCN(merge2, initializer, kernel_size_GCN = 15)
    BR1 = __BR(GCN1, initializer)
    Add1 = add([BR1, Upsample2])
    BR1_1 = __BR(Add1, initializer)
    Upsample1 = UpSampling2D(size=(2, 2), data_format=None)(BR1_1)

    
    deconv4 = Conv2DTranspose(20 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv1'+namePostFix)(merge2)

    deconv4_1 = Conv2DTranspose(20 * width, (3, 3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv1_1'+namePostFix)(merge2)
    deconv4_2 = Conv2DTranspose(20 * width, (5, 5), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv1_2'+namePostFix)(merge2)


    x_combineMask = Multiply()([x,x_mask])
    
    merge4 = Concatenate()([deconv4,x_combineMask, deconv4_1, deconv4_2])
    GCN4 = __GCN(merge4, initializer, kernel_size_GCN = 20)
    BR4 = __BR(GCN4, initializer)
    Add4 = add([BR4, Upsample1])

    BR4_1 = __BR(Add4, initializer)
    deconv5 = Conv2DTranspose(8 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv5'+namePostFix)(BR4_1)
    deconv5_1 = Conv2DTranspose(8 * width, (3,3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv5_1'+namePostFix)(BR4_1)
    deconv5_2 = Conv2DTranspose(8 * width, (5,5), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv5_2'+namePostFix)(BR4_1)

    merge5 = Concatenate()([deconv5, deconv5_1, deconv5_2])

    GCN5 = __GCN(merge5, initializer, kernel_size_GCN = 20)
    BR1_2 = __BR(GCN5, initializer)
    
    deconv6 = Conv2DTranspose(8 * width, (kernel_size_deconv, kernel_size_deconv), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv6'+namePostFix)(BR1_2)
    deconv6_1 = Conv2DTranspose(8 * width, (3,3), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv6_1'+namePostFix)(BR1_2)
    deconv6_2 = Conv2DTranspose(8 * width, (5,5), padding='same', strides = strides_num*2, activation='relu', kernel_initializer = initializer, name='deconv6_2'+namePostFix)(BR1_2)

    merge6 = Concatenate()([deconv6, deconv6_1, deconv6_2])
    
    
    BR1_3 = __BR_1(merge6, initializer)
    
    output = Activation(bound_relu(maxvalue=1.0))(BR1_3)
    
    targetAreaOutput=BinaryMaskMulLayer()([output,mask_input])
    
    holeImage=DigLayer()([img_input,mask_inputCombine])
    
    finalOutput= Add()([holeImage,targetAreaOutput])
    
    return finalOutput
       

#Snow impaint Model    

def build_SnowMultiPathImpaintModel(shape):
    print('Build SnowImpaintModel')
    img_input = Input(shape=shape)
    mask_inputB = Input(shape=shape)
    mask_inputM = Input(shape=shape)
    mask_inputS = Input(shape=shape)
    mask_inputCombine = Input(shape=shape)
    mask_inputCombine_otherThres = Input(shape=shape)
    
    
    
    xB=Deep_Desnow_Model_B(img_input,mask_inputCombine,mask_inputCombine,mask_inputCombine_otherThres,'BImpaint')
    
    #inputs = [BSnow_input,MSnow_input,SSnow_input]
    inputs = [img_input,mask_inputB,mask_inputM,mask_inputS,mask_inputCombine,mask_inputCombine_otherThres]
    # Create model.
    model = Model(inputs, [xB], name='DeSnowNet')
    return model 

    
def build_CombineModel(shape,inputThres):
    import model.model_snowatt as modelPy
    modelAtt=modelPy.build_SnowAttentionModel((480,640,3))

    #Build Desnow Model
    
    DesnowModel=build_SnowMultiPathImpaintModel(shape)
    
    #Build Combine Model
    inpImg=Input(shape=(480,640,3))
    
    snowAttB,snowAttM,snowAttS=modelAtt(inpImg)
    
    mbAttB=binary_mul_layer(3.0)(snowAttB)
    mbAttM=binary_mul_layer(2.0)(snowAttM)
    mbAttS=binary_mul_layer(1.0)(snowAttS)
    
    combineWeightMask=Maximum()([mbAttB,mbAttM,mbAttS])
    
    fAttB=AndLayer(3.0)([combineWeightMask])
    fAttM=AndLayer(2.0)([combineWeightMask])
    fAttS=AndLayer(1.0)([combineWeightMask])
    
    fAttB=binary_mul_layer(1.0)(fAttB)
    fAttM=binary_mul_layer(1.0)(fAttM)
    fAttS=binary_mul_layer(1.0)(fAttS)
    fAttB = Concatenate()([fAttB,fAttB,fAttB])
    fAttM = Concatenate()([fAttM,fAttM,fAttM])
    fAttS = Concatenate()([fAttS,fAttS,fAttS])
    
    fcombineWeightMask = binary_mul_layer(1.0)(combineWeightMask)
    fcombineWeightMask = Concatenate()([fcombineWeightMask,fcombineWeightMask,fcombineWeightMask])
    
    otherThresCombineWeightMask=Maximum()([snowAttB,snowAttM,snowAttS])
    otherThresCombineWeightMask = binary_mul_layer(1.0,inputThres)(otherThresCombineWeightMask)
    otherThresCombineWeightMask = Concatenate()([otherThresCombineWeightMask,otherThresCombineWeightMask,otherThresCombineWeightMask])
    
    DesnowOutputB=DesnowModel([inpImg,fAttB,fAttM,fAttS,fcombineWeightMask,otherThresCombineWeightMask])
    
    modelBuildGene=Model([inpImg],[DesnowOutputB])
    return modelBuildGene 

