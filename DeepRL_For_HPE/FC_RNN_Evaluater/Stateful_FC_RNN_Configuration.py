# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import datetime
from keras.optimizers import Adam
from keras.models import Sequential
from keras.preprocessing import image
from keras.applications import vgg16, nasnet, inception_v3
from keras import regularizers, Model
from keras.layers import TimeDistributed, LSTM, Dense, Dropout, Flatten
def now(): return str(datetime.datetime.now())

######## CONF_Begins_Here ##########
confFile = 'Stateful_FC_RNN_Configuration.py'
RECORD = True # False # 

output_begin = 3
num_outputs = 3

timesteps = 1 # TimeseriesGenerator Handles overlapping
learning_rate =  0.0001
in_epochs = 1
out_epochs = 1
eva_epoch = 1
train_batch_size = 1
test_batch_size = 1

subjectList = [i for i in range(1, 25)] # [1, 2, 3, 4, 5, 7, 8, 11, 12, 14] # [9] # 
testSubjects = [3, 5, 9, 14] # [9, 18, 21, 24] # [9] # 
trainingSubjects = [s for s in subjectList if not s in testSubjects] # subjectList # 

num_datasets = len(subjectList)

lstm_nodes = 20
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

def getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, 
                  lstm_recurrent_dropout = lstm_recurrent_dropout, num_outputs = num_outputs, 
                  lr = learning_rate, include_vgg_top = include_vgg_top, use_vgg16 = use_vgg16):
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
        x = cnn_model.layers[-1].output
       # x = Dropout(0.25, name = 'dropout3_025')(x)
        x = Dense(1024, activation='relu', name='fc1024')(x)
        x = Dropout(0.25, name = 'dropout_025')(x)
        x = Dense(num_outputs, name = 'fc3')(x)
        cnn_model = Model(inputs=cnn_model.input,outputs=x)
    """
    """

    #cnn_model.summary()
    rnn = Sequential()
    rnn.add(TimeDistributed(cnn_model, batch_input_shape=(train_batch_size, timesteps, inp[0], inp[1], inp[2]), name = 'tdCNN')) 
    if not include_vgg_top:
        rnn.add(TimeDistributed(Flatten()))
    """
    rnn.add(TimeDistributed(Dropout(0.25), name = 'dropout025_conv')) #kernel_regularizer=regularizers.l2(0.001)
    rnn.add(TimeDistributed(Dense(1024), name = 'fc1024')) # , activation='relu', activation='relu', 
    rnn.add(TimeDistributed(Dropout(0.25), name = 'dropout025'))
    rnn.add(TimeDistributed(Dense(num_outputs), name = 'fc3'))

    """
    rnn.add(LSTM(lstm_nodes, dropout=lstm_dropout, recurrent_dropout=lstm_recurrent_dropout, stateful=True))#, activation='relu'
    modelID = modelID + '_seqLen%d' % timesteps
    modelID = modelID + '_stateful'
    modelID = modelID + '_lstm%d' % lstm_nodes
    rnn.add(Dense(num_outputs))
    
    modelID = modelID + '_output%d' % num_outputs

    modelID = modelID + '_BatchSize%d' % train_batch_size
    modelID = modelID + '_inEpochs%d' % in_epochs
    modelID = modelID + '_outEpochs%d' % out_epochs
    
    #for layer in rnn.layers[:1]: 
    #    layer.trainable = False
    adam = Adam(lr=lr)
    modelID = modelID + '_AdamOpt_lr-%f' % lr
    rnn.compile(optimizer=adam, loss='mean_absolute_error') #'mean_squared_error', metrics=['mae'])#
    modelID = modelID + '_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    return cnn_model, rnn, modelID, preprocess_input
