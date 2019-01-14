# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.LSTM_VGG16Helper import *
else:
    from NeighborFolderimporter import *
    from LSTM_VGG16Helper import *

from DatasetHandler.BiwiBrowser import *


output_begin = 4
num_outputs = 1

timesteps = 16 # TimeseriesGenerator Handles overlapping
learning_rate = 0.0001
in_epochs = 1
out_epochs = 7
train_batch_size = 10
test_batch_size = 10

subjectList = [1, 2, 3, 4, 5, 7, 8, 11, 12, 14] #9[i for i in range(1, 9)] + [i for i in range(10, 25)] except [6, 13, 10, ]
testSubjects = [9]
trainingSubjects = [s for s in subjectList if not s in testSubjects]

num_datasets = len(subjectList)

lstm_nodes = 256
lstm_dropout=0.25
lstm_recurrent_dropout=0.25
num_outputs = num_outputs
lr=0.0001
include_vgg_top = False

angles = ['Pitch', 'Yaw', 'Roll'] 

def getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, 
                  lstm_recurrent_dropout = lstm_recurrent_dropout, num_outputs = num_outputs, 
                  lr = learning_rate, include_vgg_top = include_vgg_top):
    inp = (224, 224, 3)
    vgg_model = VGG16(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 
    
    if include_vgg_top:
        vgg_model.layers.pop()
        vgg_model.outputs = [vgg_model.layers[-1].output]
        vgg_model.output_layers = [vgg_model.layers[-1]] 
        vgg_model.layers[-1].outbound_nodes = []

    rnn = Sequential()
    rnn.add(TimeDistributed(vgg_model, input_shape=(timesteps, inp[0], inp[1], inp[2]), name = 'tdVGG16')) 
    rnn.add(TimeDistributed(Flatten()))
    """
    rnn.add(TimeDistributed(Dropout(0.25)))#
    rnn.add(TimeDistributed(Dense(4096, activation='relu'), name = 'fc1024'))#, activation='relu'
    rnn.add(TimeDistributed(Dropout(0.25)))#
    rnn.add(TimeDistributed(Dense(4096, activation='relu'), name = 'fc104'))   # 
    rnn.add(TimeDistributed(Dropout(0.25)))#
    rnn.add(TimeDistributed(Dense(1024, activation='relu'), name = 'fc10'))#
    rnn.add(TimeDistributed(Dropout(0.25)))
    """

    rnn.add(LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout))
    rnn.add(Dense(num_outputs))

    for layer in rnn.layers[:1]: 
        layer.trainable = False
    adam = optimizers.Adam(lr=lr)
    rnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    return vgg_model, rnn

def trainCNN_LSTM(full_model, out_epochs, subjectList, timesteps, output_begin, num_outputs, 
                  batch_size = train_batch_size, in_epochs = in_epochs):
    full_model = trainImageModelForEpochs(full_model, out_epochs, subjectList, timesteps, False, 
                                          output_begin, num_outputs, batch_size = batch_size, 
                                          in_epochs = in_epochs)
    return full_model

def evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, num_outputs, angles = angles):
    if num_outputs == 1: angles = ['Yaw']
    predictions = full_model.predict_generator(test_gen, verbose = 1)
    #kerasEval = full_model.evaluate_generator(test_gen)
    predictions = predictions * label_rescaling_factor
    test_labels = test_labels * label_rescaling_factor
    outputs = []
    print('For the Subject %d (%s):' % (subject, BIWI_Subject_IDs[subject]))
    for i in range(num_outputs):
        matrix = numpy.concatenate((test_labels[timesteps:, i:i+1], predictions[:, i:i+1]), axis=1)
        differences = (test_labels[timesteps:, i:i+1] - predictions[:, i:i+1])
        absolute_mean_error = np.abs(differences).mean()
        print("\tThe absolute mean error on %s angle estimation: %.2f Degree" % (angles[i], absolute_mean_error))
        outputs.append((matrix, absolute_mean_error))
    return outputs

def evaluateAverage(results, angles = angles, num_outputs = num_outputs):
    num_testSubjects = len(results)
    sums = [0] * num_outputs
    for subject, outputs in results:
        for an, (matrix, absolute_mean_error) in enumerate(outputs):
            sums[an] += absolute_mean_error
    means = [s/num_testSubjects for s in sums]
    print('On average in %d test subjects:' % num_testSubjects)
    for i, avg in enumerate(means):
        print("\tThe absolute mean error on %s angle estimations: %.2f Degree" % (angles[i], avg))
    return means

def evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                     num_outputs = num_outputs, batch_size = test_batch_size, angles = angles):
    if num_outputs == 1: angles = ['Yaw']
    test_generators, test_labelSets = getTestBiwiForImageModel(testSubjects, timesteps, False, 
                                            output_begin, num_outputs, batch_size = test_batch_size)
    results = []
    for subject, test_gen, test_labels in zip(testSubjects, test_generators, test_labelSets):
        outputs = evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, num_outputs, angles)
        results.append((subject, outputs))
    means = evaluateAverage(results, angles, num_outputs)
    return means, results 

def drawPlotsForSubj(outputs, subj, subjID, modelID, num_outputs = num_outputs, angles = angles, save = False):
    if num_outputs == 1: angles = ['Yaw']
    colors = ['#FFAA00', '#00AA00', '#0000AA', '#AA0000'] 
    title = 'Estimations for the Subject %d (Subject ID: %s, Total length: %d)\nby the Model %s' % (subj, subjID, outputs[0][0].shape[0], modelID)
    red, blue = (1.0, 0.95, 0.95), (0.95, 0.95, 1.0)
    f, rows = plt.subplots(num_outputs, 1, sharey=True, sharex=True, figsize=(16, 3*num_outputs))
    f.suptitle(title)
    for i, (matrix, absolute_mean_error) in enumerate(outputs):
        cell = rows
        if num_outputs > 1: cell = rows[i]
        l1 = cell.plot(matrix[:, 0], colors[i], label='Ground-truth')
        l2 = cell.plot(matrix[:, 1], colors[-1], label='Estimation')
        cell.set_facecolor(red if 'F' in subjID else blue)
        #cell.set_xlim([0, 1000])
        cell.set_ylim([-label_rescaling_factor, label_rescaling_factor])
        cell.set_ylabel('%s Angle\nAbsolute Mean Error: %.2f' % (angles[i], absolute_mean_error))
    f.subplots_adjust(top=0.93, hspace=0, wspace=0)
    if save:
        plt.savefig('foo.png', bbox_inches='tight')

def drawResults(outputs, modelID, num_outputs = num_outputs, angles = angles, save = False):
    for subject, outputs in results:
        drawPlotsForSubj(outputs, subject, BIWI_Subject_IDs[subject], modelID, angles = angles, save = False)

def main():
    vgg_model, full_model = getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, 
                      lstm_dropout = lstm_dropout, lstm_recurrent_dropout = lstm_recurrent_dropout, 
                      num_outputs = num_outputs, lr = learning_rate, include_vgg_top = include_vgg_top)

    full_model = trainCNN_LSTM(full_model, out_epochs, subjectList, timesteps, output_begin, num_outputs, 
                  batch_size = train_batch_size, in_epochs = in_epochs)
    
    means, results = evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size)

    drawResults(outputs, modelID, num_outputs = num_outputs, angles = angles, save = True)
    