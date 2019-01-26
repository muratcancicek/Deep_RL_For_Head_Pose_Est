# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.EvaluationRecorder import *
else:
    from NeighborFolderimporter import *
    from EvaluationRecorder import *

from DatasetHandler.BiwiBrowser import *

import keras
import random 
import numpy as np
from keras import Model 
from keras.layers import *
from keras import optimizers
import matplotlib.pyplot as plt
from keras.optimizers import SGD
from keras.models import Sequential
from keras.models import load_model
from keras.constraints import maxnorm
from keras.applications.nasnet import NASNetLarge
from keras.applications.vgg16 import VGG16
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers.convolutional import MaxPooling2D
from keras.layers.convolutional import Convolution2D
from keras.applications.vgg16 import preprocess_input
from keras.applications.vgg16 import decode_predictions
from keras.preprocessing.sequence import TimeseriesGenerator

def trainImageModelOnSets(model, epoch, trainingSubjects, set_gen, timesteps, output_begin, num_outputs, batch_size, in_epochs = 1, stateful = False, record = False):
    c = 0
    for inputMatrix, labels in set_gen:
        subj = trainingSubjects[c]
        printLog('%d. set (Dataset %d) being trained for epoch %d by %s!' % (c+1, trainingSubjects[c], epoch+1, now()), record = record)
        labels = labels[:, output_begin:output_begin+num_outputs]
        if timesteps == None:
            model.fit(inputMatrix, labels, epochs=in_epochs, verbose=1) 
        else:
            start_index = (inputMatrix.shape[0] % batch_size) - 1 if stateful else 0                
            data_gen = TimeseriesGenerator(inputMatrix, labels, length=timesteps, batch_size=batch_size, start_index=start_index)
            model.fit_generator(data_gen, steps_per_epoch=len(data_gen), epochs=in_epochs, verbose=1) 
        if stateful:
            model.reset_states()
        c += 1
    return model

def trainImageModelForEpochs(model, epochs, trainingSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, in_epochs = 1, stateful = False, record = False):
    for e in range(epochs):
        random.Random(4).shuffle(trainingSubjects)
        trainingBiwi = readBIWIDataset(subjectList = trainingSubjects) #, scaling = False, timesteps = timesteps, overlapping = overlapping
        model = trainImageModelOnSets(model, e, trainingSubjects, trainingBiwi, timesteps, output_begin, num_outputs, batch_size, in_epochs, stateful = stateful, record = record)
        printLog('Epoch %d completed!' % (e+1), record = record)
    return model

def getTestBiwiForImageModel(testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, stateful = False, record = False):
    test_generators, test_labelSets = [], [] 
    testBiwi = readBIWIDataset(subjectList = testSubjects) #, scaling = False, timesteps = timesteps, overlapping = overlapping
    for inputMatrix, labels in testBiwi:
        labels = labels[:, output_begin:output_begin+num_outputs]
        if timesteps == None:
            test_generators.append((inputMatrix, labels))
        else:
            start_index = 0
            if stateful:
                start_index = (inputMatrix.shape[0] % batch_size) - 1 if batch_size > 1 else 0
            data_gen = TimeseriesGenerator(inputMatrix, labels, length=timesteps, batch_size=batch_size, start_index=start_index)
            test_generators.append(data_gen)
            if stateful:
                labels = labels[start_index:]
        test_labelSets.append(labels)
    return test_generators, test_labelSets

def trainAngleModelOnSets(model, epoch, trainingSubjects, set_gen, timesteps, output_begin, num_outputs, batch_size, in_epochs = 1, record = False):
    c = 0
    for inputMatrix, labels in set_gen:
        printLog('%d. set (Dataset %d) being trained for epoch %d! by %s' % (c+1, trainingSubjects[c], epoch+1, now()), record = record)
        labels = labels[:, output_begin:output_begin+num_outputs]
        data_gen = TimeseriesGenerator(labels, labels, length=timesteps, batch_size=batch_size)
        model.fit_generator(data_gen, steps_per_epoch=len(data_gen), epochs=in_epochs, verbose=1) 
        c += 1
    return model

def trainAngleModelForEpochs(model, epochs, trainingSubjects, testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, in_epochs = 1, record = False):
    for e in range(epochs):
        random.Random(4).shuffle(trainingSubjects)
        trainingBiwi = readBIWIDataset(subjectList = trainingSubjects) #, timesteps = timesteps, overlapping = overlapping
        model = trainAngleModelOnSets(model, e, trainingSubjects, trainingBiwi, timesteps, output_begin, num_outputs, batch_size, in_epochs, record = record)
        printLog('Epoch %d completed!' % (e+1), record = record)
    return model

def getTestBiwiForAngleModel(testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size):
    test_generators, test_labelSets = [], [] 
    testBiwi = readBIWIDataset(subjectList = testSubjects) #, timesteps = timesteps, overlapping = overlapping
    for inputMatrix, labels in testBiwi:
        labels = labels[:, output_begin:output_begin+num_outputs]
        data_gen = TimeseriesGenerator(labels, labels, length=timesteps, batch_size=batch_size)
        test_generators.append(data_gen)
        test_labelSets.append(labels)
    return test_generators, test_labelSets

def combined_generator(inputMatrix, labels, timesteps, batch_size):
    img_gen = TimeseriesGenerator(inputMatrix, labels, length=timesteps, batch_size=batch_size)
    ang_gen = TimeseriesGenerator(labels, labels, length=timesteps, batch_size=batch_size)
    for (inputMatrix, outputLabels0), (inputLabels, outputLabels) in zip(img_gen, ang_gen):
        yield [inputMatrix, inputLabels], outputLabels
            

def trainFinalModelOnSets(model, epoch, trainingSubjects, set_gen, timesteps, output_begin, num_outputs, batch_size, in_epochs = 1, record = False):
    c = 0
    for inputMatrix, labels in set_gen:
        printLog('%d. set (Dataset %d) being trained for epoch %d!' % (c+1, trainingSubjects[c], epoch+1), record = record)
        labels = labels[:, output_begin:output_begin+num_outputs]
        data_gen = combined_generator(inputMatrix, labels, timesteps, batch_size)
        model.fit_generator(data_gen, steps_per_epoch=len(labels), epochs=in_epochs, verbose=1) 
        c += 1
    return model

def trainFinalModelForEpochs(model, epochs, trainingSubjects, testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, in_epochs = 1, record = False):
    for e in range(epochs):
        random.Random(4).shuffle(trainingSubjects)
        trainingBiwi = readBIWIDataset(subjectList = trainingSubjects) #, timesteps = timesteps, overlapping = overlapping
        model = trainFinalModelOnSets(model, e, trainingSubjects, trainingBiwi, timesteps, output_begin, num_outputs, batch_size, in_epochs, record = record)
        printLog('Epoch %d completed!' % (e+1), record = record)
    return model

def getTestBiwiForFinalModel(testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, record = False):
    test_generators, test_labelSets = [], [] 
    testBiwi = readBIWIDataset(subjectList = testSubjects) #, timesteps = timesteps, overlapping = overlapping
    for inputMatrix, labels in testBiwi:
        labels = labels[:, output_begin:output_begin+num_outputs]
        data_gen = combined_generator(inputMatrix, labels, timesteps, batch_size)
        test_generators.append(data_gen)
        test_labelSets.append(labels)
    return test_generators, test_labelSets

if __name__ == "__main__":
    printLog('Done')