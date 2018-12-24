# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *

from DatasetHandler.BiwiBrowser import *
import keras
import numpy as np
#from keras import Model 
from keras.layers import *
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.constraints import maxnorm
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions

def scale(arr):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler = scaler.fit(arr)
    # normalize the dataset and printscaler, 
    normalized = scaler.transform(arr)
    return normalized 

def reshaper(m, l, timesteps = 1):
    wasted = (m.shape[0] % timesteps)
    m, l = m[wasted:], l[wasted:]
    l = scale(l)
    m = m.reshape((int(m.shape[0]/timesteps), timesteps, m.shape[1], m.shape[2], m.shape[3]))
    l = l.reshape((int(l.shape[0]/timesteps), timesteps, l.shape[1]))
    l = l[:, -1, :]
    return m, l

num_datasets = 2

num_outputs = 1

timesteps = 1

#keras.backend.clear_session()
def getFinalModel(num_outputs = num_outputs):
    dense_layer_1 = 1#int((patch_size[0] * patch_size[1]) / 1)0010#00000
    dense_layer_2 = 8
    inp = BIWI_Frame_Shape
    #vgg_model = VGG16(weights='imagenet', include_top=False, input_shape = BIWI_Frame_Shape)
    rnn = Sequential()
    #rnn.add(TimeDistributed(vgg_model, batch_size = timesteps, input_shape=(timesteps, inp[0], inp[1], inp[2])))#
    
    rnn.add(TimeDistributed(Flatten(),input_shape=(timesteps, inp[0], inp[1], inp[2])))
    rnn.add(LSTM(64)) # , stateful=True, dropout=0.2, recurrent_dropout=0.2, activation='relu'
#    rnn.add(TimeDistributed(Dropout(0.2)))
    rnn.add(Dense(num_outputs))
    print('WARNING')

    for layer in rnn.layers[:15]:
        layer.trainable = False
    rnn.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return rnn

def test():
    full_model = getFinalModel(num_outputs = num_outputs)
    biwi = readBIWIDataset(subjectList = [s for s in range(1, num_datasets+1)])#
    c = 0
    frames, labelsList = [], []
    for inputMatrix, labels in biwi:
        inputMatrix, labels = reshaper(inputMatrix, labels, timesteps = timesteps)
        if c < num_datasets-1:
            full_model.fit(inputMatrix, labels[:, :num_outputs], batch_size = timesteps, nb_epoch=1, verbose=2, shuffle=False) #
            full_model.reset_states()
            frames.append(inputMatrix)
            labelsList.append(scale(labels))
        else:
            frames.append(inputMatrix)
            labelsList.append(scale(labels))
        c += 1
        print('Batch %d done!' % c)

    test_inputMatrix, test_labels = frames[0], labelsList[0]

    predictions = full_model.predict(test_inputMatrix, batch_size = timesteps)

    output1 = numpy.concatenate((test_labels[:, :1], predictions[:, :1]), axis=1)

    plt.plot(output1)
    plt.show()
    
if __name__ == "__main__":
    test()
    print('Done')