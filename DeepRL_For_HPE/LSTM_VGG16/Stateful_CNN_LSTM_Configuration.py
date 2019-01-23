# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.LSTM_VGG16Helper import *
else:
    from NeighborFolderimporter import *
    from LSTM_VGG16Helper import *

RECORD = False # True # 

output_begin = 3
num_outputs = 3

timesteps = 1 # TimeseriesGenerator Handles overlapping
learning_rate = 0.0001
in_epochs = 1
out_epochs = 20
train_batch_size = 5
test_batch_size = 4

subjectList = [9] # [i for i in range(1, 25)] # [1, 2, 3, 4, 5, 7, 8, 11, 12, 14]  # 
testSubjects = [9] # [9, 18, 21, 24] # 
trainingSubjects = subjectList # [s for s in subjectList if not s in testSubjects] #

num_datasets = len(subjectList)

lstm_nodes = 320
lstm_dropout=0.25
lstm_recurrent_dropout=0.25
include_vgg_top = False

angles = ['Pitch', 'Yaw', 'Roll'] 


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
    rnn.add(TimeDistributed(vgg_model, batch_input_shape=(inp[0], inp[1], inp[2]), name = 'tdVGG16')) #timesteps, input_shape
    rnn.add(TimeDistributed(Flatten()))
    """
    rnn.add(TimeDistributed(Dropout(0.25)))#
    rnn.add(TimeDistributed(Dense(4096, activation='relu'), name = 'fc1024'))#, activation='relu'
    rnn.add(TimeDistributed(Dropout(0.25)))#
    rnn.add(TimeDistributed(Dense(4096, activation='relu'), name = 'fc104'))   # 
    rnn.add(TimeDistributed(Dropout(0.25)))#
    rnn.add(TimeDistributed(Dense(1024, activation='relu'), name = 'fc10'))#
    """
    rnn.add(TimeDistributed(Dropout(0.25)))

    rnn.add(LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, stateful=True))
    modelID = modelID + '_seqLen%d' % timesteps
    modelID = modelID + '_stateful'
    modelID = modelID + '_lstm%d' % lstm_nodes
    rnn.add(Dense(num_outputs))
    
    modelID = modelID + '_output%d' % num_outputs

    modelID = modelID + '_inEpochs%d' % in_epochs
    modelID = modelID + '_outEpochs%d' % out_epochs
    
    for layer in rnn.layers[:1]: 
        layer.trainable = False
    adam = optimizers.Adam(lr=lr)
    modelID = modelID + '_AdamOpt_lr-%f' % lr
    rnn.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae'])
    modelID = modelID + '_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    return vgg_model, rnn, modelID
