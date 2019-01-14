from DatasetHandler.BiwiBrowser import *
from LSTM_VGG16Helper import *


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
trainingSubjects = [s for s in trainingSubjects if not s in testSubjects]

num_datasets = len(subjectList)

lstm_nodes = 256
lstm_dropout=0.25
lstm_recurrent_dropout=0.25
num_outputs = num_outputs
lr=0.0001
include_vgg_top = False

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

def evaluateSubject(subject, test_gen, test_labels, timesteps, num_outputs, angles):
    predictions = full_model.predict_generator(test_gen, verbose = 1)
    #kerasEval = full_model.evaluate_generator(test_gen)
    predictions = predictions * label_rescaling_factor
    test_labels = test_labels * label_rescaling_factor
    outputs = []
    print('For the Subject %d:' % subject)
    for i in num_outputs:
        matrix = numpy.concatenate((test_labels[timesteps:, i:i+1], predictions[:, i:i+1]), axis=1)
        differences = (test_labels[timesteps:, i:i+1] - predictions[:, i:i+1])
        absolute_mean_error = np.abs(differences).mean()
        print("\tThe absolute mean error on %s angle estimation: %.2f Degree" % (angles[i], absolute_mean_error))
        outputs.append((matrix, absolute_mean_error))
    return outputs

def evaluateAverage(results):
    num_testSubjects = len(results)
    sums = [0, 0, 0]
    for subject, outputs in results:
        for an, (matrix, absolute_mean_error) in enumerate(outputs):
            sums[an] += absolute_mean_error
    means = [s/num_testSubjects for s in sums]
    print('On average in %d test subjects:' % num_testSubjects)
    for avg in means:
        print("\tThe absolute mean error on %s angle estimations: %.2f Degree" % (angles[i], avg))
    return means

def evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size):
    test_generators, test_labelSets = getTestBiwiForImageModel(testSubjects, timesteps, False, 
                                            output_begin, num_outputs, batch_size = test_batch_size)
    results = []
    angles = ['Pitch', 'Yaw', 'Roll'] if num_outputs > 1 else ['Yaw']
    for subject, test_gen, test_labels in zip(testSubjects, test_generators, test_labelSets):
        outputs = evaluateSubject(subject, test_gen, test_labels, timesteps, num_outputs, angles)
        results.append((subject, outputs))
    means = evaluateAverage(results)
    return means, results 


def drawPlotsForSubj(outputs, subj, subjID, num_outputs = num_outputs):
    colors, angles = ['r', 'b', 'g'], ['Pitch', 'Yaw', 'Roll'] if num_outputs > 1 else ['Yaw']
    titles[0] = 'Estimations for the Subject %d (SubjID: %s, Total length: %d\n)' % (subj, subjectIDs[row], labels.shape[0])
    red, blue = (1.0, 0.95, 0.95), (0.95, 0.95, 1.0)
    f, rows = plt.subplots(num_outputs, 1, sharex=True, figsize=(16, 4*num_outputs))
    for col in range(num_outputs):
        cell = rows
        if len(labelSets) > 1: cell = rows[row][col] if num_cols > 1 else rows[row]
        cell.plot(labels[:, num_outputs+col+cs], colors[col+cs])
        cell.set_facecolor(red if 'F' in subjectIDs[row] else blue)
        cell.set_xlim([0, 1000])
        cell.set_ylim([-90, 90])
        if len(labelSets) > 1: left_cell = rows[row][0] if num_cols > 1 else rows[row]
        left_cell.set_ylabel()
    row += 1


    f.subplots_adjust(hspace=0, wspace=0)
    #plt.setp([a.get_yticklabels() for a in f.axes[1:]], visible=False)

def evaluate(results, num_outputs = num_outputs):
    #for labels in labelSets:
    for subject, outputs in results:
        p

    plt.figure(figsize=(17,5*num_outputs))
    plt.plot(output1)

def main():
    vgg_model, full_model = getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, 
                      lstm_dropout = lstm_dropout, lstm_recurrent_dropout = lstm_recurrent_dropout, 
                      num_outputs = num_outputs, lr = learning_rate, include_vgg_top = include_vgg_top)

    means, results = evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size)
