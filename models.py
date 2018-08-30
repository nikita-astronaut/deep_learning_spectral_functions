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
    model = Sequential()
    model.add(Conv1D(filters=512, kernel_size=3, input_shape=(input_length, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dropout(0.4))
    model.add(Dense(2048, activation='relu'))
    model.add(Dense(1024, activation='relu'))
    model.add(Dense(output_length))

    model.compile(optimizer=optimizer, loss='mse')

    return model
