# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *
from keras.models import load_model
import shutil
import os

importNeighborFolders()
from paths import *

sep = os.sep # '/' # 
dir_path = os.path.dirname(os.path.realpath(__file__))
outputFolder = 'results'
CURRENT_MODEL = 'Last_Model' 

def saveConfiguration(confFile = None, record = True):
    if confFile == None: confFile = 'Stateful_CNN_LSTM_Configuration.py'
    with open(confFile, 'r') as conf:
        save = False
        for line in conf:
            line = line[:-1]
            if line == '######## CONF_Begins_Here ##########':
                save = True
            if save:
                printLog(line, record = record)
            if line == '######### CONF_ends_Here ###########':
                save = False

def getModelFolder(modelID, create = True):  
    modelID = outputFolder + sep + modelID
    if create and not os.path.isdir(modelID):
        os.mkdir(modelID)
    return modelID

def addModelFolder(modelID, name, create = True): 
    return getModelFolder(modelID, create) + sep + name

def printLog(*args, **kwargs):
    record = kwargs.pop('record', False)
    print(*args, **kwargs)
    if not record: return 
    fName = addModelFolder(CURRENT_MODEL, 'output_%s.txt' % CURRENT_MODEL)
    with open(fName, 'a') as file:
        print(*args, **kwargs, file=file)

def loadKerasModel(modelID, record = True):
    fileName = '%s.h5' % (modelID)
    model = load_model(Keras_Models_Folder + fileName)
    printLog('%s has been saved.' % fileName, record = record)
    return model
        
def saveLast_Model(last_Model):
    if os.path.isdir(last_Model):
        tempName = last_Model + '_'
        saveLast_Model(tempName)
        os.rename(last_Model, tempName)

def startRecording(modelID, record = True):
    if record:
        last_Model = getModelFolder(CURRENT_MODEL, create = False)
        saveLast_Model(last_Model)
    printLog('Model %s has been started to be evaluated.' % modelID, record = record)
        
def saveKerasModel(model, modelID, record = True):
    if record:
        fileName = '%s.h5' % (modelID)
        model.save(Keras_Models_Folder + sep + fileName)
        printLog('%s has been saved.' % fileName, record = record)

def completeRecording(modelID, record = True, interrupt = False):
        
    if not interrupt:
        printLog('Model %s has been evaluated successfully.' % modelID, record = record)
    if record:
        fName = addModelFolder(CURRENT_MODEL, 'output_%s.txt' % CURRENT_MODEL, create = False)
        newName = addModelFolder(CURRENT_MODEL, 'output_%s.txt' % modelID, create = False)
        os.rename(fName, newName)
        os.rename(getModelFolder(CURRENT_MODEL, create = False), getModelFolder(modelID, False))
        print('Model %s has been recorded successfully.' % modelID)
    
def interruptSmoothly(full_model, modelID, record = True):
    print()
    printLog('Model %s has been interrupted.' % modelID, record = record)
    saveKerasModel(full_model, modelID, record = record)
    completeRecording(modelID, record = record, interrupt = True)
    printLog('Terminating...', record = record)
    exit()

if __name__ == "__main__":
    startRecording('modelID')
    saveConfiguration()
    completeRecording('modelID', True)
