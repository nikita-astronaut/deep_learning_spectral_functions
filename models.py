from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Input, Flatten, Add, GlobalMaxPooling2D
from keras.layers import Concatenate, concatenate
from keras.layers import Activation, LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam, RMSprop, SGD, Adadelta

def dense_model(input_length, output_length):
    optimizer = Adam(lr = 0.001, decay=0.0)

    input_bands = Input(shape=[input_length], name = 'correlator')
    dense_1 = Dense(32)(input_bands)
    dense_1 = Activation('elu')(dense_1)
    dense_1 = Dropout(0.3)(dense_1, training=True)

    dense_2 = Dense(64)(dense_1)
    dense_2 = Activation('elu')(dense_2)
    dense_2 = Dropout(0.3)(dense_2, training=True)

    dense_3 = Dense(32)(dense_2)
    dense_3 = Activation('elu')(dense_3)
    dense_3 = Dropout(0.3)(dense_3, training=True)

    output = Dense(output_length)(dense_3)
    output = Activation('sigmoid')(output)

    model = Model(inputs=[input_bands], outputs=output)

    model.compile(optimizer=optimizer, loss='mse')

    return model
