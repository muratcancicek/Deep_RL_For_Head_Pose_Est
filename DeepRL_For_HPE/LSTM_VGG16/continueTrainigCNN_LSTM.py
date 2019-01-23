# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
STATEFUL = True
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from LSTM_VGG16.LSTM_VGG16Helper import *
    from LSTM_VGG16.EvaluationRecorder import *
    from LSTM_VGG16.runCNN_LSTM import *
    if STATEFUL:
        from LSTM_VGG16.Stateful_CNN_LSTM_Configuration import *
    else:
        from LSTM_VGG16.CNN_LSTM_Configuration import *
else:
    from NeighborFolderimporter import *
    from LSTM_VGG16Helper import *
    from EvaluationRecorder import *
    from runCNN_LSTM import *
    if STATEFUL:
        from Stateful_CNN_LSTM_Configuration import *
    else:
        from CNN_LSTM_Configuration import *

importNeighborFolders()
from DatasetHandler.BiwiBrowser import *

if len(sys.argv) != 2:
    print('Needs modelID argument. Try again...')
    exit()
    
modelID = sys.argv[1]

in_epochs = 1
out_epochs = 5

def continueTrainigCNN_LSTM(record = False, modelID = modelID):
    full_model = loadKerasModel(modelID, record = record) 
    modelStr = modelID
    modelID = 'Exp' + modelID[-19:]
    extension = '_and_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    modelID = modelID + extension
    modelStr = modelStr + extension
    
    printLog(get_model_summary(full_model), record = record)
    
    if True:
        print('Training model %s' % modelStr)
        full_model = trainCNN_LSTM(full_model, modelID, out_epochs, trainingSubjects, timesteps, output_begin, num_outputs, 
                      batch_size = train_batch_size, in_epochs = in_epochs, record = record)
        if not ((out_epochs + in_epochs + num_datasets) < 10):
            saveKerasModel(full_model, modelID, record = record)
        
    printLog('The subjects are trained:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    
    printLog('Evaluating model %s' % modelStr, record = record)
    printLog('The subjects will be tested:', [(s, BIWI_Subject_IDs[s]) for s in testSubjects], record = record)
    means, results = evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size, stateful = STATEFUL, record = record)

    figures = drawResults(results, modelStr, modelID, num_outputs = num_outputs, angles = angles, save = record)    
    
    completeRecording(modelID, record = record)

def main():
    continueTrainigCNN_LSTM(record = RECORD)

if __name__ == "__main__":
    main()
    print('Done')