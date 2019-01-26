# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import os, numpy, random
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from FC_RNN_Evaluater.EvaluationRecorder import *
else:
    from NeighborFolderimporter import *
    from EvaluationRecorder import *

from DatasetHandler.BiwiBrowser import readBIWIDataset, BIWI_Subject_IDs, now, label_rescaling_factor


from keras.preprocessing.sequence import TimeseriesGenerator

######### Training Methods ###########
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

def trainImageModelForEpochs(model, epochs, trainingSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, in_epochs = 1, stateful = False, record = False, preprocess_input = None):
    for e in range(epochs):
        random.Random(4).shuffle(trainingSubjects)
        trainingBiwi = readBIWIDataset(subjectList = trainingSubjects, preprocess_input = preprocess_input) #, scaling = False, timesteps = timesteps, overlapping = overlapping
        model = trainImageModelOnSets(model, e, trainingSubjects, trainingBiwi, timesteps, output_begin, num_outputs, batch_size, in_epochs, stateful = stateful, record = record)
        printLog('Epoch %d completed!' % (e+1), record = record)
    return model

def trainCNN_LSTM(full_model, modelID, out_epochs, subjectList, timesteps, output_begin, num_outputs, 
                  batch_size, in_epochs, stateful = False, record = False, preprocess_input = None):
    try:
        full_model = trainImageModelForEpochs(full_model, out_epochs, subjectList, timesteps, False, 
                                          output_begin, num_outputs, batch_size = batch_size, 
                                          in_epochs = in_epochs, stateful = stateful, record = record, preprocess_input = preprocess_input)
    except KeyboardInterrupt:
        interruptSmoothly(full_model, modelID, record = record)
    return full_model

######### Evaluation Methods ###########
def getTestBiwiForImageModel(testSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, stateful = False, record = False, preprocess_input = None):
    test_generators, test_labelSets = [], [] 
    testBiwi = readBIWIDataset(subjectList = testSubjects, preprocess_input = preprocess_input) #, scaling = False, timesteps = timesteps, overlapping = overlapping
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

def evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, num_outputs, angles, batch_size, stateful = False, record = False):
    if num_outputs == 1: angles = ['Yaw']
    printLog('For the Subject %d (%s):' % (subject, BIWI_Subject_IDs[subject]), record = record)
    predictions = full_model.predict_generator(test_gen, steps = int(len(test_labels)/batch_size), verbose = 1)
    full_model.reset_states()
    #kerasEval = full_model.evaluate_generator(test_gen)
    predictions = predictions * label_rescaling_factor
    test_labels = test_labels * label_rescaling_factor
    outputs = []
    for i in range(num_outputs):
        if stateful:
            start_index = (test_labels.shape[0] % batch_size) if batch_size > 1 else 0
            matrix = numpy.concatenate((test_labels[start_index:, i:i+1], predictions[:, i:i+1]), axis=1)
            differences = (test_labels[start_index:, i:i+1] - predictions[:, i:i+1])
        else:
            print(test_labels[:, i:i+1].shape, predictions[:, i:i+1].shape)
            matrix = numpy.concatenate((test_labels[:, i:i+1], predictions[:, i:i+1]), axis=1)
            differences = (test_labels[:, i:i+1] - predictions[:, i:i+1])
        absolute_mean_error = np.abs(differences).mean()
        printLog("\tThe absolute mean error on %s angle estimation: %.2f Degree" % (angles[i], absolute_mean_error), record = record)
        outputs.append((matrix, absolute_mean_error))
    return full_model, outputs

def evaluateAverage(results, angles, num_outputs, record = False):
    num_testSubjects = len(results)
    sums = [0] * num_outputs
    for subject, outputs in results:
        for an, (matrix, absolute_mean_error) in enumerate(outputs):
            sums[an] += absolute_mean_error
    means = [s/num_testSubjects for s in sums]
    printLog('On average in %d test subjects:' % num_testSubjects, record = record)
    for i, avg in enumerate(means):
        printLog("\tThe absolute mean error on %s angle estimations: %.2f Degree" % (angles[i], avg), record = record)
    return means

def evaluateCNN_LSTM(full_model, label_rescaling_factor, testSubjects, timesteps, output_begin, 
                     num_outputs, batch_size, angles, stateful = False, record = False, preprocess_input = None):
    if num_outputs == 1: angles = ['Yaw']
    test_generators, test_labelSets = getTestBiwiForImageModel(testSubjects, timesteps, False, output_begin, num_outputs, 
                                            batch_size = batch_size, stateful = stateful, record = record, preprocess_input = preprocess_input)
    results = []
    for subject, test_gen, test_labels in zip(testSubjects, test_generators, test_labelSets):
        full_model, outputs = evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, num_outputs, angles, batch_size = batch_size, stateful = stateful, record = record)
        results.append((subject, outputs))
    means = evaluateAverage(results, angles, num_outputs, record = record)
    return full_model, means, results 

if __name__ == "__main__":
    print('Done')