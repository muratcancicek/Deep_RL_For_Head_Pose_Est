# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
#MODEL_TYPE = 0 # Stateful LSTM # 1 # Stateless LSTM # 2 CNN
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from CNN_Evaluater.CNN_Evaluater import *
else:
    from CNN_Evaluater import *
from FC_RNN_Evaluater.EvaluationRecorder import *
from FC_RNN_Evaluater.FC_RNN_Evaluater import *
from FC_RNN_Evaluater.runFC_RNN_Experiment import *
from FC_RNN_Evaluater.EstimationPlotter import *
from CNN_Evaluater import *
from CNN_Configuration import *

def runCNN_ExperimentWithModel(full_model, modelID, modelStr, in_epochs, exp = -1, record = False, preprocess_input = None):
    print('Training model %s' % modelStr)
    full_model = trainCNN(full_model, modelID, in_epochs, trainingSubjects, output_begin, num_outputs, 
                  batch_size = train_batch_size, record = record, preprocess_input = preprocess_input)
    #if not ((out_epochs + in_epochs + num_datasets) < 10):
    #    saveKerasModel(full_model, modelID, record = record)        
    
    printLog('The subjects are trained:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    printLog('Evaluating model %s' % modelStr, record = record)
    printLog('The subjects will be tested:', [(s, BIWI_Subject_IDs[s]) for s in testSubjects], record = record)
    full_model, means, results = evaluateCNN(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, output_begin = output_begin, num_outputs = num_outputs, 
                    batch_size = test_batch_size, angles = angles, record = record, preprocess_input = preprocess_input)
    return full_model, means, results

def runCNN_Evaluater(record = False):
    vgg_model, full_model, modelID, preprocess_input = getFinalModel(num_outputs = num_outputs, lr = learning_rate, include_vgg_top = include_vgg_top, use_vgg16 = use_vgg16)
    modelStr = modelID
    modelID = 'Exp' + modelStr[-19:]
    startRecording(modelID, record = record)
    printLog(get_model_summary(vgg_model), record = record)
    printLog(get_model_summary(full_model), record = record)
    saveConfiguration(confFile = confFile, record = record)
    for e in range(out_epochs):
        full_model, means, results = runCNN_ExperimentWithModel(full_model, modelID, modelStr, eva_epoch, record = record, preprocess_input = preprocess_input)
        printLog('%s completed!' % (modelID), record = record)
        printLog('Experiment %d completed!' % (e+1), record = record)
    if not ((out_epochs + in_epochs + num_datasets) < 10):
        saveKerasModel(full_model, modelID, record = record)        
   
    figures = drawResults(results, modelStr, modelID, num_outputs = num_outputs, angles = angles, save = record)         
    completeRecording(modelID, record = record)

def main():
    runCNN_Evaluater(record = RECORD)

if __name__ == "__main__":
    main()
    print('Done')