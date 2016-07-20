from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from extra_layers import LRNLayer
#from keras.layers.core import MaskedLayer

def keras_LAP(cls='fc1',drop=0):
    '''
    Builds the keras version of the LAP age model. 

    INPUT: (optional) layer to pop, percentage to drop
    OUTPUT: Keras model function to be compiled
    '''
    model = Sequential()

    # Layer 1 conv1_1 (3)
    model.add(ZeroPadding2D((1, 1), input_shape=(3, 224, 224)))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 2 conv1_2 (6)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 3 pool1 (7)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 4 conv2_1 (10)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 5 conv2_2 (13)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 6 pool2 (14)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 7 conv3_1 (17)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 8 conv3_2 (20)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 9 conv3_3 (23)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 10 pool3 (24)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 11 conv4_1 (27)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 12 conv4_2 (30)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 13 conv4_3 (33)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 14 pool4 (34)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 15 conv5_1 (37)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 16 conv5_2 (40)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, subsample=(1,1)))
    model.add(Activation('relu'))

    # Layer 17 conv5_3 (43)
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(512, 3, 3, activation='relu'))
    model.add(Activation('relu'))

    # Layer 18 pool5 (44)
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='valid'))

    # Layer 19 fc6 - INNER PRODUCT (48)
    model.add(Flatten())
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(drop))

    # Layer 12 fc7 - INNER PRODUCT (51)
    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(drop))

    # Layer 19 fc8-101 - INNER PRODUCT
    model.add(Dense(101))
    model.add(Activation('softmax'))

    return model

