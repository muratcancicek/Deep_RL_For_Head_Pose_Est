# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
import io
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.LSTM_VGG16Helper import *
    from LSTM_VGG16.EvaluationRecorder import *
else:
    from NeighborFolderimporter import *
    from LSTM_VGG16Helper import *
    from EvaluationRecorder import *

from DatasetHandler.BiwiBrowser import *

RECORD = True # False #

output_begin = 3
num_outputs = 3

timesteps = 1 # TimeseriesGenerator Handles overlapping
learning_rate = 0.0001
in_epochs = 1
out_epochs = 1
train_batch_size = 5
test_batch_size = 4

subjectList = [9]#[i for i in range(1, 25)] [1, 2, 3, 4, 5, 7, 8, 11, 12, 14][9]#9[i for i in range(1, 9)] +  except [6, 13, 10, ]
testSubjects = [1]# [9, 18, 21, 24]
trainingSubjects = [s for s in subjectList if not s in testSubjects]

num_datasets = len(subjectList)

lstm_nodes = 26
lstm_dropout=0.25
lstm_recurrent_dropout=0.25
num_outputs = num_outputs
lr=0.0001
include_vgg_top = False

angles = ['Pitch', 'Yaw', 'Roll'] 

def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, 
                  lstm_recurrent_dropout = lstm_recurrent_dropout, num_outputs = num_outputs, 
                  lr = learning_rate, include_vgg_top = include_vgg_top):
    modelID = 'VGG16' 
    inp = (224, 224, 3)
    vgg_model = VGG16(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 
    
    if include_vgg_top:
        modelID = modelID + '_inc_top'
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
    modelID = modelID + '_seqLen%d' % timesteps
    modelID = modelID + '_lstm%d' % lstm_nodes
    rnn.add(Dense(num_outputs))
    
    modelID = modelID + '_output%d' % num_outputs

    for layer in rnn.layers[:1]: 
        layer.trainable = False
    adam = optimizers.Adam(lr=lr)
    modelID = modelID + '_AdamOpt(lr=%f)' % lr
    rnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    modelID = modelID + '_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    return vgg_model, rnn, modelID

def trainCNN_LSTM(full_model, out_epochs, subjectList, timesteps, output_begin, num_outputs, 
                  batch_size = train_batch_size, in_epochs = in_epochs, record = False):
    full_model = trainImageModelForEpochs(full_model, out_epochs, subjectList, timesteps, False, 
                                          output_begin, num_outputs, batch_size = batch_size, 
                                          in_epochs = in_epochs, record = record)
    return full_model

def evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, num_outputs, angles = angles, record = False):
    if num_outputs == 1: angles = ['Yaw']
    printLog('For the Subject %d (%s):' % (subject, BIWI_Subject_IDs[subject]), record = record)
    predictions = full_model.predict_generator(test_gen, verbose = 1)
    #kerasEval = full_model.evaluate_generator(test_gen)
    predictions = predictions * label_rescaling_factor
    test_labels = test_labels * label_rescaling_factor
    outputs = []
    for i in range(num_outputs):
        matrix = numpy.concatenate((test_labels[timesteps:, i:i+1], predictions[:, i:i+1]), axis=1)
        differences = (test_labels[timesteps:, i:i+1] - predictions[:, i:i+1])
        absolute_mean_error = np.abs(differences).mean()
        printLog("\tThe absolute mean error on %s angle estimation: %.2f Degree" % (angles[i], absolute_mean_error), record = record)
        outputs.append((matrix, absolute_mean_error))
    return outputs

def evaluateAverage(results, angles = angles, num_outputs = num_outputs, record = False):
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

def evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                     num_outputs = num_outputs, batch_size = test_batch_size, angles = angles, record = False):
    if num_outputs == 1: angles = ['Yaw']
    test_generators, test_labelSets = getTestBiwiForImageModel(testSubjects, timesteps, False, 
                                            output_begin, num_outputs, batch_size = test_batch_size, record = record)
    results = []
    for subject, test_gen, test_labels in zip(testSubjects, test_generators, test_labelSets):
        outputs = evaluateSubject(full_model, subject, test_gen, test_labels, timesteps, num_outputs, angles, record = record)
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
    return f

def drawResults(results, modelID, num_outputs = num_outputs, angles = angles, save = False):
    figures = []
    for subject, outputs in results:
        f = drawPlotsForSubj(outputs, subject, BIWI_Subject_IDs[subject], modelID, angles = angles, save = False)
        figures.append((f, subject))
    if save:
        for f, subj in figures:
            fileName = 'subject%d_%s.png' % (subj, modelID)
            f.savefig(addModelFolder(CURRENT_MODEL, fileName), bbox_inches='tight')
            printLog(fileName, 'has been saved by %s.' % now(), record = save)
    return figures

def runCNN_LSTM(record = False):
    vgg_model, full_model, modelID = getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, 
                      lstm_dropout = lstm_dropout, lstm_recurrent_dropout = lstm_recurrent_dropout, 
                      num_outputs = num_outputs, lr = learning_rate, include_vgg_top = include_vgg_top)
    startRecording(modelID, record = record)
    printLog(get_model_summary(vgg_model), record = record)
    printLog(get_model_summary(full_model), record = record)
    if record:
        fileName = '%s.h5' % (modelID)
        full_model.save(addModelFolder(CURRENT_MODEL, fileName))
    print('Training model %s' % modelID)
    full_model = trainCNN_LSTM(full_model, out_epochs, trainingSubjects, timesteps, output_begin, num_outputs, 
                  batch_size = train_batch_size, in_epochs = in_epochs, record = record)
    printLog('The subjects are trained:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    printLog('Evaluating model %s' % modelID, record = record)
    printLog('The subjects will be tested:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    means, results = evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size, record = record)

    figures = drawResults(results, modelID, num_outputs = num_outputs, angles = angles, save = record)
    

    completeRecording(modelID, record = record)

def main():
    runCNN_LSTM(record = RECORD)

if __name__ == "__main__":
    main()
    print('Done')