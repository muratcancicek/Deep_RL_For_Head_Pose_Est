# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
#MODEL_TYPE = 0 # Stateful LSTM # 1 # Stateless LSTM # 2 CNN
STATEFUL = True # False # None # 
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from FC_RNN_Evaluater.FC_RNN_Evaluater import *
    from FC_RNN_Evaluater.EvaluationRecorder import *
    if STATEFUL:
        from FC_RNN_Evaluater.Stateful_FC_RNN_Configuration import *
    else:
        from FC_RNN_Evaluater.Stateless_FC_RNN_Configuration import *
else:
    from Core.NeighborFolderimporter import *
    if STATEFUL:
        from Stateful_FC_RNN_Configuration import *
    else:
        from Stateless_FC_RNN_Configuration import *
    from EstimationPlotter import *
    from FC_RNN_Evaluater import *

import io
def get_model_summary(model):
    stream = io.StringIO()
    model.summary(print_fn=lambda x: stream.write(x + '\n'))
    summary_string = stream.getvalue()
    stream.close()
    return summary_string

def runCNN_LSTM_ExperimentWithModel(full_model, modelID, modelStr, out_epochs, exp = -1, record = False, preprocess_input = None):
    print('Training model %s' % modelStr)
    full_model = trainCNN_LSTM(full_model, modelID, out_epochs, trainingSubjects, timesteps, output_begin, num_outputs, 
                  batch_size = train_batch_size, in_epochs = in_epochs, stateful = STATEFUL, exp = exp, record = record, preprocess_input = preprocess_input)
    if not ((out_epochs + in_epochs + num_datasets) < 10):
        saveKerasModel(full_model, modelID, record = record)        
    
    printLog('The subjects are trained:', [(s, BIWI_Subject_IDs[s]) for s in trainingSubjects], record = record)
    printLog('Evaluating model %s' % modelStr, record = record)
    printLog('The subjects will be tested:', [(s, BIWI_Subject_IDs[s]) for s in testSubjects], record = record)
    full_model, means, results = evaluateCNN(full_model, label_rescaling_factor = label_rescaling_factor, 
                     testSubjects = testSubjects, timesteps = timesteps,  output_begin = output_begin, 
                    num_outputs = num_outputs, batch_size = test_batch_size, angles = angles, stateful = STATEFUL, record = record, preprocess_input = preprocess_input)
    return full_model, means, results

def runCNN_LSTM(record = False):
    vgg_model, full_model, modelID, preprocess_input = getFinalModel(timesteps = timesteps, lstm_nodes = lstm_nodes, lstm_dropout = lstm_dropout, lstm_recurrent_dropout = lstm_recurrent_dropout, 
                      num_outputs = num_outputs, lr = learning_rate, include_vgg_top = include_vgg_top, use_vgg16 = use_vgg16)
    modelStr = modelID
    modelID = 'Exp' + modelStr[-19:]
    startRecording(modelID, record = record)
    printLog(get_model_summary(vgg_model), record = record)
    printLog(get_model_summary(full_model), record = record)
    saveConfiguration(confFile = confFile, record = record)

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
   
    figures = drawResults(results, modelStr, modelID, num_outputs = num_outputs, angles = angles, save = record)         
    completeRecording(modelID, record = record)

def main():
    runCNN_LSTM(record = RECORD)

if __name__ == "__main__":
    main()
    print('Done')