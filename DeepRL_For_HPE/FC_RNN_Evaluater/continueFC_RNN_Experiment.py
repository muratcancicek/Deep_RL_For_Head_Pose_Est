# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
STATEFUL = True
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from FC_RNN_Evaluater.FC_RNN_Evaluater import *
    from FC_RNN_Evaluater.EvaluationRecorder import *
    from FC_RNN_Evaluater.runFC_RNN_Experiment import *
    if STATEFUL:
        from FC_RNN_Evaluater.Stateful_FC_RNN_Configuration import *
    else:
        from FC_RNN_Evaluater.FC_RNN_Configuration import *
else:
    from NeighborFolderimporter import *
    from FC_RNN_Evaluater import *
    from EvaluationRecorder import *
    from runFC_RNN_Experiment import *
    if STATEFUL:
        from Stateful_FC_RNN_Configuration import *
    else:
        from Stateful_FC_RNN_Configuration import *

importNeighborFolders()
from DatasetHandler.BiwiBrowser import *
import keras

if not len(sys.argv) in [2, 3]:
    print('Needs modelID argument. Try again...')
    exit()
      
modelID = sys.argv[1]
trainMore = True
if len(sys.argv) == 2:
    trainMore = False
elif not sys.argv[2] in ['trainMore', 'evaluateOnly']:
    print('Incorrect argument for method. Try again...')
    exit()   
elif sys.argv[2] == 'evaluateOnly':
    trainMore = False
    
def continueTrainigCNN_LSTM(record = False, modelID = modelID):
    vgg_model, fake_model, fake_modelID, preprocess_input = getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, lstm_recurrent_dropout = lstm_recurrent_dropout, 
                      num_outputs = num_outputs, lr = learning_rate, include_vgg_top = include_vgg_top)
    full_model = loadKerasModel(modelID, record = record) 
    modelStr = modelID
    modelID = 'Exp' + modelID[-19:]
    extension = '_and_%s' % now()[:-7].replace(' ', '_').replace(':', '-')
    modelID = modelID + extension
    modelStr = modelStr + extension
    
    printLog(get_model_summary(full_model), record = record)
    saveConfiguration(confFile = confFile, record = record)
    
    if trainMore:
        # keras.backend.clear_session() 
        num_experiments = int(out_epochs / eva_epoch) if out_epochs > eva_epoch else 1
        for exp in range(1, num_experiments+1):
            modelID = 'Exp' + modelStr[-19:] + '_part%d' % exp
            full_model, means, results = runCNN_LSTM_ExperimentWithModel(full_model, modelID, modelStr, eva_epoch, exp = exp, record = record, preprocess_input = preprocess_input)
            printLog('%s completed!' % (modelID), record = record)

        if out_epochs % eva_epoch > 0 and num_experiments > 1:
            modelID = 'Exp' + modelStr[-19:] + '_part%d' % exp
            full_model, means, results = runCNN_LSTM_ExperimentWithModel(full_model, modelID, modelStr, out_epochs % eva_epoch, exp = exp, record = record, preprocess_input = preprocess_input)
            printLog('%s completed!' % (modelID), record = record)
        modelID = 'Exp' + modelStr[-19:]
        saveKerasModel(full_model, modelID, record = record)   
        
    printLog('The subjects are trained:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    
    printLog('Evaluating model %s' % modelStr, record = record)
    printLog('The subjects will be tested:', [(s, BIWI_Subject_IDs[s]) for s in testSubjects], record = record)
    full_model, means, results = evaluateCNN_LSTM(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps, output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size, stateful = STATEFUL, angles = angles, record = record)

    figures = drawResults(results, modelStr, modelID, num_outputs = num_outputs, angles = angles, save = record)    
    
    completeRecording(modelID, record = record)

def main():
    continueTrainigCNN_LSTM(record = RECORD)

if __name__ == "__main__":
    main()
    print('Done')