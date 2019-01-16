# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
import sys
import io
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.LSTM_VGG16Helper import *
    from LSTM_VGG16.EvaluationRecorder import *
    from LSTM_VGG16.runCNN_LSTM import *
else:
    from NeighborFolderimporter import *
    from LSTM_VGG16Helper import *
    from EvaluationRecorder import *
    from LSTM_VGG16.runCNN_LSTM import *

importNeighborFolders()
from DatasetHandler.BiwiBrowser import *

if len(sys.argv) != 2:
    print('Needs modelID argument. Try again...')
    exit()
    
modelID = sys.argv[1]

in_epochs = 7
out_epochs = 1

def continueTrainigCNN_LSTM(record = False, modelID = modelID):
    full_model = loadKerasModel(modelID, record = record) 
    modelID = modelID + '_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    
    startRecording(modelID, record = record)
    printLog(get_model_summary(full_model), record = record)
    
    print('Training model %s' % modelID)
    full_model = trainCNN_LSTM(full_model, modelID, out_epochs, trainingSubjects, timesteps, output_begin, num_outputs, 
                  batch_size = train_batch_size, in_epochs = in_epochs, record = record)
    if not ((out_epochs + in_epochs + num_datasets) < 10):
        saveKerasModel(full_model, modelID, record = record)
        
    printLog('The subjects are trained:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    
    printLog('Evaluating model %s' % modelID, record = record)
    printLog('The subjects will be tested:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    means, results = evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size, record = record)

    figures = drawResults(results, modelID, num_outputs = num_outputs, angles = angles, save = record)
    
    completeRecording(modelID, record = record)

def main():
    continueTrainigCNN_LSTM(record = RECORD)

if __name__ == "__main__":
    main()
    print('Done')