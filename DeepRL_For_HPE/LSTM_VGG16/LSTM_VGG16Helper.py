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
from random import shuffle
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
from keras.preprocessing.sequence import TimeseriesGenerator

def trainOnSets(trainingSubjects, epochs, model, set_gen, timesteps, output_begin, num_outputs, batch_size, in_epochs = 1):
    c = 0
    for inputMatrix, labels in set_gen:
        print('%d. set (Dataset %d) being trained for epoch %d!' % (c+1, trainingSubjects[c], epoch+1))
        labels = labels[:, output_begin:output_begin+num_outputs]
        data_gen = TimeseriesGenerator(inputMatrix, labels, length=timesteps, batch_size=batch_size)
        model.fit_generator(data_gen, steps_per_epoch=len(data_gen), epochs=in_epochs, verbose=1) 
        c += 1
    return model

def trainForEpochs(model, epochs, subjectList, testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, in_epochs = 1):
    trainingSubjects = [s for s in subjectList if s not in testSubjects]
    for e in range(epochs):
        shuffle(trainingSubjects)
        trainingBiwi = readBIWIDataset(subjectList = trainingSubjects) #, timesteps = timesteps, overlapping = overlapping
        model = trainOnSets(model, trainingSubjects, trainingBiwi, timesteps, output_begin, num_outputs, batch_size, in_epochs)
        print('Epoch %d completed!' % (e+1))
    return model

def getTestBiwi(testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size):
    test_generators, test_labelSets = [], [] 
    testBiwi = readBIWIDataset(subjectList = testSubjects) #, timesteps = timesteps, overlapping = overlapping
    for inputMatrix, labels in testBiwi:
        labels = labels[:, output_begin:output_begin+num_outputs]
        data_gen = TimeseriesGenerator(inputMatrix, labels, length=timesteps, batch_size=batch_size)
        test_generators.append(data_gen)
        test_labelSets.append(labels)
    return test_generators, test_labelSets

if __name__ == "__main__":
    print('Done')