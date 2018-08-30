from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Add, GlobalMaxPooling2D
from keras.layers import Concatenate, concatenate, Conv1D
from keras.layers import Activation, LeakyReLU, Convolution1D
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD, Adadelta

def dense_model(input_length, output_length):
    optimizer = Adam(lr = 0.001)

    input_bands = Input(shape=[input_length], name = 'correlator')
    dense_1 = Dense(256)(input_bands)
    dense_1 = Activation('relu')(dense_1)
    dense_1 = Dropout(0.1)(dense_1, training=True)

    dense_2 = Dense(512)(dense_1)
    dense_2 = Activation('relu')(dense_2)
    dense_2 = Dropout(0.1)(dense_2, training=True)

    # dense_3 = Dense(256)(dense_2)
    # dense_3 = Activation('relu')(dense_3)
    # dense_3 = Dropout(0.1)(dense_3, training=True)

    output = Dense(output_length)(dense_2)
    # output = Activation('sigmoid')(output)

    model = Model(inputs=[input_bands], outputs=output)

    model.compile(optimizer=optimizer, loss='mse')

    return model

def conv_model(input_length, output_length):
    optimizer = Adam(lr = 0.001)
    input_bands = Input(shape=(input_length, 1))
    conv1 = Conv1D(filters=512, kernel_size=3, input_shape=(input_length, 1))(input_bands)
    conv1 = Activation('relu')(conv1)
    conv1 = Flatten()(conv1)
    conv1 = Dropout(0.4)(conv1, training=True)
    
    dense = Dense(2048)(conv1)
    dense = Dense(1024)(dense)
    output = Dense(output_length)(dense)    
    model = Model(inputs=[input_bands], outputs=output)
    model.compile(optimizer=optimizer, loss='mse')

    return model
