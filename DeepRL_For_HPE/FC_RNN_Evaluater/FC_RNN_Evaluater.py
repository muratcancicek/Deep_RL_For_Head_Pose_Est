# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import os, numpy, random
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from FC_RNN_Evaluater.EvaluationRecorder import *
    from FC_RNN_Evaluater.ReinforceAlgorithmForKerasModels import reinforceModel
else:
    from NeighborFolderimporter import *
    from EvaluationRecorder import *
    from ReinforceAlgorithmForKerasModels import reinforceModel

from DatasetHandler.BiwiBrowser import readBIWIDataset, BIWI_Subject_IDs, now, label_rescaling_factor, BIWI_Lebel_Scalers, unscaleAnnoByScalers

from keras.preprocessing.sequence import TimeseriesGenerator
        
######### Training Methods ###########
def combined_generator(inputMatrix, labels, timesteps, batch_size):
    img_gen = TimeseriesGenerator(inputMatrix[1:], labels[:-1], length=timesteps, batch_size=batch_size)
    ang_gen = TimeseriesGenerator(labels, labels, length=timesteps, batch_size=batch_size)
    for (inputMatrix, outputLabels0), (inputLabels, outputLabels) in zip(img_gen, ang_gen):
        yield [inputMatrix, inputLabels], outputLabels
        
def getSequencesToSequences(inputMatrix, labels, timesteps):
    offset = labels.shape[0]%timesteps
    if offset > 0: offset = timesteps - offset
    npad = ((offset+1, timesteps), (0, 0), (0, 0), (0, 0))
    inputMatrix = np.pad(inputMatrix[1:], pad_width=npad, mode='constant', constant_values=0)
    npad = ((offset+1, timesteps), (0, 0))
    inputLabels = np.pad(labels[:-1], pad_width=npad, mode='constant', constant_values=0)
    npad = ((timesteps+offset+1, 0), (0, 0))
    targets = np.pad(labels[1:], pad_width=npad, mode='constant', constant_values=0)
    outputLabels = np.zeros(targets.shape[:1]+(timesteps,)+targets.shape[1:])
    il = np.zeros_like(inputLabels)
    for i in range(0, inputLabels.shape[0], timesteps):
        il[i] = inputLabels[i]
    inputLabels = il
    for i in range(0, len(targets), timesteps):
        outputLabels[i] = targets[i:i+timesteps] 
    return inputMatrix, inputLabels, outputLabels

def combined_generator2(inputMatrix, inputLabels, outputLabels, timesteps, batch_size):   
    img_gen = TimeseriesGenerator(inputMatrix, outputLabels, length=timesteps, stride=timesteps, batch_size=batch_size)
    ang_gen = TimeseriesGenerator(inputLabels, outputLabels, length=timesteps, stride=timesteps, batch_size=batch_size)
    for (inputMatrix, outputLabels0), (inputLabels, outputLabels) in zip(img_gen, ang_gen):
        yield [inputMatrix, inputLabels], outputLabels
            
def trainImageModelOnSets(model, epoch, trainingSubjects, set_gen, timesteps, output_begin, num_outputs, batch_size, episodes = 5, sigma = 0.05, in_epochs = 1, stateful = False, exp = -1, record = False):
    c = 0
    for inputMatrix, labels in set_gen:
        subj = trainingSubjects[c]
        expStr = ' in Experiment %d' % (exp) if exp != -1 else '' 
        printLog('%d. set (Dataset %d) being trained for epoch %d%s by %s!' % (c+1, trainingSubjects[c], epoch+1, expStr, now()), record = record)
        labels = labels[:, output_begin:output_begin+num_outputs]
        if timesteps == None:
            model.fit(inputMatrix, labels, epochs=in_epochs, verbose=1) 
        else:
            start_index = (inputMatrix.shape[0] % batch_size) - 1 if stateful else 0     
            #data_gen = TimeseriesGenerator(inputMatrix[1:], labels[:-1], length=timesteps, batch_size=batch_size, start_index=start_index)
            inputMatrix, inputLabels, outputLabels = getSequencesToSequences(inputMatrix, labels, timesteps) 
            data_gen = combined_generator2(inputMatrix, inputLabels, outputLabels, timesteps, batch_size)
            steps_per_epoch = ((inputMatrix.shape[0]/timesteps)-1)/batch_size
            #model.fit_generator(data_gen, steps_per_epoch = steps_per_epoch, epochs=in_epochs, verbose=1) 
            model = reinforceModel(model, data_gen, episodes, sigma, steps_per_epoch, in_epochs, verbose=1)
        if stateful:  model.reset_states()
        c += 1
    return model

def trainImageModelForEpochs(model, epochs, trainingSubjects, timesteps, overlapping, output_begin, num_outputs, batch_size, episodes = 5, sigma = 0.05, in_epochs = 1, stateful = False, exp = -1, record = False, preprocess_input = None):
    for e in range(epochs):
        random.Random(4).shuffle(trainingSubjects)
        trainingBiwi = readBIWIDataset(subjectList = trainingSubjects, preprocess_input = preprocess_input) #, scaling = False, timesteps = timesteps, overlapping = overlapping
        model = trainImageModelOnSets(model, e, trainingSubjects, trainingBiwi, timesteps, output_begin, num_outputs, batch_size, episodes, sigma, in_epochs, exp = exp, stateful = stateful, record = record)
        expStr = ' for Experiment %d' % (exp) if exp != -1 else '' 
        printLog('Epoch %d%s completed!' % (e+1, expStr), record = record)
    return model

def trainCNN_LSTM(full_model, modelID, out_epochs, subjectList, timesteps, output_begin, num_outputs, 
                  batch_size, in_epochs, episodes = 5, sigma = 0.05, exp = -1, stateful = False, record = False, preprocess_input = None):
    try:
        full_model = trainImageModelForEpochs(full_model, out_epochs, subjectList, timesteps, False, 
                                          output_begin, num_outputs, batch_size = batch_size, episodes = episodes, sigma = sigma, 
                                          in_epochs = in_epochs, stateful = stateful, exp = exp, record = record, preprocess_input = preprocess_input)
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
            data_gen = TimeseriesGenerator(inputMatrix[1:], labels[:-1], length=timesteps, batch_size=batch_size, start_index=start_index)
            test_generators.append(data_gen)
            if stateful:
                labels = labels[start_index:]
        test_labelSets.append(labels)
    return test_generators, test_labelSets

def unscaleEstimations(test_labels, predictions, scalers, output_begin, num_outputs):
    """* label_rescaling_factor * label_rescaling_factor
    """
    sclrs = [scalers[0][output_begin:output_begin+num_outputs], scalers[1][output_begin:output_begin+num_outputs]]
    test_labels = unscaleAnnoByScalers(test_labels, sclrs)
    predictions = unscaleAnnoByScalers(predictions, sclrs)
    return test_labels, predictions

def evaluateOutputsForSubject(test_labels, predictions, timesteps, output_begin, num_outputs, angles, batch_size, stateful = False, record = False):
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
    total = 0
    for m, avg in outputs: total += avg
    printLog("\tThe absolute mean error on average: %.2f Degree" % (total/num_outputs), record = record)
    return outputs

def evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size, stateful = False, record = False):
    if num_outputs == 1: angles = ['Yaw']
    printLog('For the Subject %d (%s):' % (subject, BIWI_Subject_IDs[subject]), record = record)
    predictions = full_model.predict_generator(test_gen, steps = int(len(test_labels)/batch_size), verbose = 1)
    if stateful:  full_model.reset_states()
    test_labels, predictions = unscaleEstimations(test_labels, predictions, BIWI_Lebel_Scalers, output_begin, num_outputs)
    #kerasEval = full_model.evaluate_generator(test_gen) 
    outputs = evaluateOutputsForSubject(test_labels, predictions, timesteps, output_begin, num_outputs, angles, batch_size, stateful, record)
    return full_model, outputs

def slide(m, x):
        m[0, :-1] = m[0, 1:]
        m[0, -1] = x
        return m

def predicter(full_model, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size):
    cur_pred = np.zeros((len(test_labels)+1, num_outputs))
    #pred = []
    c = 0
    printProgressBar(c, len(test_labels), prefix = 'Estimating...', suffix = 'Complete', length = 50)
    for (inputMatrix, inputLabels) in test_gen:
        c+=1
        printProgressBar(c, len(test_labels), prefix = 'Estimating...', suffix = 'Complete', length = 50)
        if c > len(test_labels): break
        p = full_model.predict([inputMatrix, cur_pred[c-1].reshape((batch_size, timesteps, num_outputs))])
        #print(p, cur_pred[c-1], inputLabels.reshape((batch_size, timesteps, num_outputs)))
        cur_pred[c] = p
    #print(cur_pred)
    return cur_pred[1:]

def predicter2(full_model, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size):
    inputMatrix, inputLabels = test_gen
    inputMatrix, inputLabels, outputLabels = getSequencesToSequences(inputMatrix, inputLabels, timesteps) 
    predictions = np.zeros((inputMatrix.shape[0], num_outputs))
    data_gen = combined_generator2(inputMatrix, inputLabels, outputLabels, timesteps, batch_size)
    i = 0
    printProgressBar(i, len(inputLabels), prefix = 'Estimating...', suffix = 'Complete', length = 50)
    for (inputMatrix, inputLabels), outputLabels in data_gen:
        for inputSequence, inputLabels, outputLabels in zip(inputMatrix, inputLabels, outputLabels):
            predictions[i:i+timesteps] = full_model.predict([inputMatrix, np.zeros_like(inputLabels[np.newaxis, ...])])
            #])predictions[i-timesteps:i]
            i += timesteps
            printProgressBar(i, len(inputLabels), prefix = 'Estimating...', suffix = 'Complete', length = 50)
    return predictions[predictions.shape[0]-test_labels.shape[0]:]
            
    
def evaluateSubjectForMultipleInput(full_model, subject, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size, stateful = False, record = False):
    if num_outputs == 1: angles = ['Yaw']
    printLog('For the Subject %d (%s):' % (subject, BIWI_Subject_IDs[subject]), record = record)
    predictions = predicter2(full_model, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size) 
    if stateful:  full_model.reset_states()
    test_labels, predictions = unscaleEstimations(test_labels, predictions, BIWI_Lebel_Scalers, output_begin, num_outputs)
    #kerasEval = full_model.evaluate_generator(test_gen) 
    outputs = evaluateOutputsForSubject(test_labels, predictions, timesteps, output_begin, num_outputs, angles, batch_size, stateful, record)
    return full_model, outputs

def evaluateAverage(results, angles, num_outputs, record = False):
    num_testSubjects = len(results)
    sums = [0] * num_outputs
    for subject, outputs in results:
        for an, (matrix, absolute_mean_error) in enumerate(outputs):
            sums[an] += absolute_mean_error
    means = [s/num_testSubjects for s in sums]
    if num_testSubjects == 1: return means
    printLog('On average in %d test subjects:' % num_testSubjects, record = record)
    for i, avg in enumerate(means):
        printLog("\tThe absolute mean error on %s angle estimations: %.2f Degree" % (angles[i], avg), record = record)
    return means

def evaluateCNN_LSTM(full_model, label_rescaling_factor, testSubjects, timesteps, output_begin, 
                     num_outputs, batch_size, angles, stateful = False, record = False, preprocess_input = None):
    if num_outputs == 1: angles = ['Yaw']
    test_generators, test_labelSets = getTestBiwiForImageModel(testSubjects, None, False, output_begin, num_outputs, 
                                            batch_size = batch_size, stateful = stateful, record = record, preprocess_input = preprocess_input)
    results = []
    for subject, test_gen, test_labels in zip(testSubjects, test_generators, test_labelSets):
        #full_model, outputs = evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size, stateful = stateful, record = record)
        full_model, outputs = evaluateSubjectForMultipleInput(full_model, subject, test_gen, test_labels, timesteps, output_begin, num_outputs, angles, batch_size = batch_size, stateful = stateful, record = record)
        results.append((subject, outputs))
    means = evaluateAverage(results, angles, num_outputs, record = record)
    return full_model, means, results 

if __name__ == "__main__":
    print('Done')