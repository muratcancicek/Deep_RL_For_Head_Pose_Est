# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import datetime
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications import vgg16, nasnet, inception_v3
from keras import regularizers, Model
from keras.layers import concatenate, Concatenate, TimeDistributed, LSTM, Dense, Dropout, Flatten, CuDNNLSTM, Input 
from Reinforce_with_Keras.ReinforcementModel import ReinforcementModel
from Reinforce_with_Keras.AdamOptimizerForRL import AdamForRL
def now(): return str(datetime.datetime.now())

######## CONF_Begins_Here ##########
confFile = 'Stateful_FC_RNN_Configuration.py'
RECORD = False # True # 

output_begin = 3
num_outputs = 3

reinforcement_episodes = 5
sampling_variance = 0.01

timesteps = 10 # TimeseriesGenerator Handles overlapping
learning_rate =  0.000001
in_epochs = 1
out_epochs = 1
eva_epoch = 1
train_batch_size = 1
test_batch_size = 1

subjectList = [14] # [1, 2, 3, 4, 5, 7, 8, 11, 12, 14] # [i for i in range(1, 25)] # 
testSubjects = [14] # [6, 9, 14, 24] # [9, 18, 21, 24] # 
trainingSubjects = subjectList # [s for s in subjectList if not s in testSubjects] # 

num_datasets = len(subjectList)

lstm_nodes = 1024
lstm_dropout = 0.25
lstm_recurrent_dropout = 0.25
include_vgg_top = True # False # 

angles = ['Pitch', 'Yaw', 'Roll'] 
use_vgg16 = True # False # 
######### CONF_ends_Here ###########

def preprocess_input_for_model(imagePath, Target_Frame_Shape, m, modelPackage):
    img = image.load_img(imagePath, target_size = Target_Frame_Shape)
    x = image.img_to_array(img)
    x = x[m[0]:-m[1], m[2]:-m[3], :]
    return modelPackage.preprocess_input(x)

def addDropout(model):
    # Store the fully connected layers
    fc1 = model.layers[-2]
    fc2 = model.layers[-2]
    predictions = model.layers[-1]

    # Reconnect the layers
    x = Dropout(0.85)(fc1.output)
    x = fc2(x)
    x = Dropout(0.85)(x)
    predictors = predictions(x)
    return Model(inputs=model.input, outputs=predictors)

def getCNN_Model(use_vgg16 = use_vgg16):
    if use_vgg16:
        modelID = 'VGG16' 
        inp = (224, 224, 3)
        modelPackage = vgg16
        margins = (8, 8, 48, 48)
        Target_Frame_Shape = (240, 320, 3)
        cnn_model = vgg16.VGG16(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 
    elif True:
        inp = (299, 299, 3)
        modelPackage = inception_v3
        modelID = 'InceptionV3'     
        margins = (0, 1, 51, 50)
        Target_Frame_Shape = (300, 400, 3)
        cnn_model = inception_v3.InceptionV3(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 
    else:
        inp = (331, 331, 3)
        modelPackage = nasnet
        modelID = 'NASNetLarge'     
        margins = (14, 15, 74, 75)
        Target_Frame_Shape = (360, 480, 3)
        cnn_model = nasnet.NASNetLarge(weights='imagenet', input_shape = inp, include_top=include_vgg_top) 

    def preprocess_input(imagePath): return preprocess_input_for_model(imagePath, Target_Frame_Shape, margins, modelPackage)
    
    if include_vgg_top:
        modelID = modelID + '_inc_top'
        #cnn_model = addDropout(cnn_model)
        cnn_model.layers.pop()
        cnn_model.outputs = [cnn_model.layers[-1].output]
        cnn_model.output_layers = [cnn_model.layers[-1]] 
        cnn_model.layers[-1].outbound_nodes = []
        for layer in cnn_model.layers: 
            layer.trainable = False
        x = cnn_model.layers[-1].output #
        """
        x = Dropout(0.25, name = 'dropout3_025')(x) #
        x = Dense(1024, activation='relu', name='fc1024')(x) #
        x = Dropout(0.25, name = 'dropout_025')(x) #
        x = Dense(num_outputs, name = 'fc3')(x) #
        """

        #a = Input(shape=(num_outputs, ), name='aux_input0')
        
        #x = (concatenate([x, a], axis = 1))#

        cnn_model = Model(inputs=cnn_model.input, outputs=x)
    return inp, cnn_model, modelID, preprocess_input

def getLSTM_Model(inp, cnn_model, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, 
                  lstm_recurrent_dropout = lstm_recurrent_dropout):
    rnn = Sequential()
    rnn.add(TimeDistributed(cnn_model, batch_input_shape=(train_batch_size, timesteps, inp[0], inp[1], inp[2]), name = 'tdCNN')) 
    if not include_vgg_top:
        rnn.add(TimeDistributed(Flatten())) 

     #rnn.add(CuDNNLSTM(lstm_nodes, stateful=True))
    rnn.add(LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, stateful=True))#, activation='relu'
    return rnn

def getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, 
                  lstm_recurrent_dropout = lstm_recurrent_dropout, num_outputs = num_outputs, 
                  lr = learning_rate, include_vgg_top = include_vgg_top, use_vgg16 = use_vgg16):

    inp, cnn_model, modelID, preprocess_input = getCNN_Model(use_vgg16 = use_vgg16)

    #finalModel = getLSTM_Model(inp, cnn_model, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, lstm_recurrent_dropout = lstm_recurrent_dropout)
    #finalModel.add(Dense(num_outputs))

    fcRNN = Sequential()
    fcRNN.add(TimeDistributed(cnn_model, batch_input_shape=(train_batch_size, timesteps, inp[0], inp[1], inp[2]), name = 'tdCNN')) 
    
    a = Input(batch_shape=(train_batch_size, timesteps, num_outputs), name='aux_input')

    x = (concatenate([fcRNN.output, a], axis = 2))

    lstm_out = LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, return_sequences=True, stateful=True)(x)
   # main_output = Dense(num_outputs)(lstm_out) # 
    main_output = TimeDistributed(Dense(num_outputs))(lstm_out) #,
    finalModel = ReinforcementModel(inputs=[fcRNN.input, a], outputs=main_output)

    #finalModel = Sequential()
    #finalModel.add(finalModel1)
    #finalModel.add(TimeDistributed(cnn_model, name = 'tdCNN')) 
    #finalModel.add(LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, stateful=True))#, activation='relu'
    #finalModel.add(Dense(num_outputs))

    adam = AdamForRL(lr=lr)
    finalModel.compile(optimizer=adam, loss='mean_squared_error', metrics=['mae']) #)# 'mean_absolute_error'

    modelID = modelID + '_seqLen%d' % timesteps; modelID = modelID + '_stateful'; modelID = modelID + '_lstm%d' % lstm_nodes
    modelID = modelID + '_output%d' % num_outputs; modelID = modelID + '_BatchSize%d' % train_batch_size
    modelID = modelID + '_inEpochs%d' % in_epochs; modelID = modelID + '_outEpochs%d' % out_epochs
    modelID = modelID + '_AdamOpt_lr-%f' % lr; modelID = modelID + '_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    
    return cnn_model, finalModel, modelID, preprocess_input
