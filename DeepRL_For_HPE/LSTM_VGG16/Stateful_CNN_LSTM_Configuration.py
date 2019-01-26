# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.LSTM_VGG16Helper import *
else:
    from NeighborFolderimporter import *
    from LSTM_VGG16Helper import *

######## CONF_Begins_Here ##########
confFile = 'Stateful_CNN_LSTM_Configuration.py'
RECORD = True # False # 

output_begin = 3
num_outputs = 3

timesteps = 1 # TimeseriesGenerator Handles overlapping
learning_rate =  0.0001
in_epochs = 1
out_epochs = 2
eva_epoch = 1
train_batch_size = 1
test_batch_size = 1

subjectList = [i for i in range(1, 25)] # [1, 2, 3, 4, 5, 7, 8, 11, 12, 14] # [9] # 
testSubjects = [3, 5, 9, 14] # [9, 18, 21, 24] # [1] # 
trainingSubjects = [s for s in subjectList if not s in testSubjects] # subjectList # 

num_datasets = len(subjectList)

lstm_nodes = 10
lstm_dropout = 0.25
lstm_recurrent_dropout = 0.25
include_vgg_top = True 

angles = ['Pitch', 'Yaw', 'Roll'] 
######### CONF_ends_Here ###########

def getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, 
                  lstm_recurrent_dropout = lstm_recurrent_dropout, num_outputs = num_outputs, 
                  lr = learning_rate, include_vgg_top = include_vgg_top, vgg16 = True):
    if vgg16:
        inp = (224, 224, 3)
        cnn_model = VGG16(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 
        modelID = 'VGG16' 
    else:
        inp = (331, 331, 3)
        cnn_model = NASNetLarge(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 
        modelID = 'NASNetLarge'     
    
    if include_vgg_top:
        modelID = modelID + '_inc_top'
        cnn_model.layers.pop()
        cnn_model.outputs = [cnn_model.layers[-1].output]
        cnn_model.output_layers = [cnn_model.layers[-1]] 
        cnn_model.layers[-1].outbound_nodes = []

    #cnn_model.summary()
    rnn = Sequential()
    rnn.add(TimeDistributed(cnn_model, batch_input_shape=(train_batch_size, timesteps, inp[0], inp[1], inp[2]), name = 'tdCNN')) 
    #rnn.add(TimeDistributed(Dropout(0.25), name = 'dropout025_conv'))
    #rnn.add(TimeDistributed(Dense(1024, activation='relu'), name = 'fc1024')) # , activation='relu'
    #rnn.add(TimeDistributed(Dropout(0.25), name = 'dropout025'))
    rnn.add(TimeDistributed(Dense(num_outputs), name = 'fc3'))

    rnn.add(LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, stateful=True))
    modelID = modelID + '_seqLen%d' % timesteps
    modelID = modelID + '_stateful'
    modelID = modelID + '_lstm%d' % lstm_nodes
    rnn.add(Dense(num_outputs))
    
    modelID = modelID + '_output%d' % num_outputs

    modelID = modelID + '_BatchSize%d' % train_batch_size
    modelID = modelID + '_inEpochs%d' % in_epochs
    modelID = modelID + '_outEpochs%d' % out_epochs
    
    for layer in rnn.layers[:1]: 
        layer.trainable = False
    adam = optimizers.Adam(lr=lr)
    modelID = modelID + '_AdamOpt_lr-%f' % lr
    rnn.compile(optimizer=adam, loss='mean_absolute_error') #'mean_squared_error', metrics=['mae']
    modelID = modelID + '_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    return cnn_model, rnn, modelID
