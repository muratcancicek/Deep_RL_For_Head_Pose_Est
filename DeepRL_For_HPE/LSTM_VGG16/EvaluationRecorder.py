import shutil
import os

sep = os.sep # '/' # 
dir_path = os.path.dirname(os.path.realpath(__file__))
outputFolder = 'results'

def getModelFolder(modelID, create = True):  
    modelID = outputFolder + sep + modelID
    if create and not os.path.isdir(modelID):
        os.mkdir(modelID)
    return modelID

def addModelFolder(modelID, name, create = True): 
    return getModelFolder(modelID, create) + sep + name

CURRENT_MODEL = 'Last_Model' 

def printLog(*args, **kwargs):
    record = kwargs.pop('record', False)
    print(*args, **kwargs)
    if not record: return 
    fName = addModelFolder(CURRENT_MODEL, 'output_%s.txt' % CURRENT_MODEL)
    with open(fName, 'a') as file:
        print(*args, **kwargs, file=file)
        
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
        
def completeRecording(modelID, record = True):
    printLog('Model %s has been evaluated successfully.' % modelID, record = record)
    if record:
        fName = addModelFolder(CURRENT_MODEL, 'output_%s.txt' % CURRENT_MODEL, create = False)
        newName = addModelFolder(CURRENT_MODEL, 'output_%s.txt' % modelID, create = False)
        os.rename(fName, newName)
        os.rename(getModelFolder(CURRENT_MODEL, create = False), getModelFolder(modelID, False))
    print('Model %s has been recorded successfully.' % modelID)
    
if __name__ == "__main__":
    startRecording('modelID')
    completeRecording('modelID', True)
