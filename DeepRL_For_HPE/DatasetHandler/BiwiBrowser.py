# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
# Dirty importing that allows the main author to switch environments easily

if 'COMPUTERNAME' in os.environ:
    if os.environ['COMPUTERNAME'] == "MSI3":
        os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

if '.' in __name__:
    from DatasetHandler.NeighborFolderimporter import *
    from DatasetHandler.BiwiTarBrowser import *
else:
    from NeighborFolderimporter import *
    from BiwiTarBrowser import *

from keras.applications.nasnet import preprocess_input
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import MinMaxScaler, scale
from keras.preprocessing.image import img_to_array
from keras.preprocessing import image
from matplotlib import pyplot
from os import listdir
import datetime
import tarfile
import cv2
import struct
import numpy
import png
import os

importNeighborFolders()
from paths import *

#################### Constants ####################
pwd = os.path.abspath(os.path.dirname(__file__))
BIWI_Data_folder = BIWI_Main_Folder + 'hpdb/'
BIWI_SnippedData_folder = pwd + '/BIWI_Files/BIWI_Samples/hpdb/'.replace('/', os.path.sep)
BIWI_Lebels_file = BIWI_Main_Folder + 'db_annotations.tgz'
BIWI_Lebels_file_Local = pwd + '/BIWI_Files/db_annotations.tgz'.replace('/', os.path.sep)
BIWI_Frame_Shape = (360, 480, 3)
BIWI_Frame_Shape = (240, 320, 3)
def now(): return str(datetime.datetime.now())
label_rescaling_factor = 90
BIWI_Subject_IDs = ['XX', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'F03', 'M09', 'M10', 'F05', 'M11', 'M12', 'F02', 'M01', 'M13', 'M14']
#################### Frame Reading ####################
def getRGBpngFileName(subject, frame):
    return str(subject).zfill(2) + '/frame_' + str(frame).zfill(5) + '_rgb.png'

def pngObjToNpArr(imagePath):
    img = image.load_img(imagePath, target_size=BIWI_Frame_Shape)
    #x = image.img_to_array(img)[14:-15, 74:-75, :]
    x = image.img_to_array(img)[8:-8, 48:-48, :]
    x = preprocess_input(x)
    return x

def getBIWIFrameAsNpArr(subject, frame, dataFolder = BIWI_Data_folder):
    imagePath = dataFolder + getRGBpngFileName(subject, frame)
    return pngObjToNpArr(imagePath)

def filterFrameNamesForSubj(subject, dataFolder):
    subjectFolder = str(subject).zfill(2) + os.path.sep
    allNames = os.listdir(dataFolder + subjectFolder)
    frameNamesForSubj = (fn for fn in allNames if '_rgb.png' in fn)
    frameKey = lambda n: str(subject).zfill(2) + '/' + n[:-8]
    absolutePath = lambda n: dataFolder + subjectFolder + n
    frameNamesForSubj = ((frameKey(n), absolutePath(n)) for n in sorted(frameNamesForSubj))
    return frameNamesForSubj

def getAllFramesForSubj(subject, dataFolder = BIWI_Data_folder):
    frameNamesForSubj = filterFrameNamesForSubj(subject, dataFolder)
    #print('Subject ' + str(subject).zfill(2) + '\'s frames have been started to read ' + now())
    frames = ((n, pngObjToNpArr(framePath)) for n, framePath in frameNamesForSubj)
    #print('Subject ' + str(subject).zfill(2) + '\'s all frames have been read by ' + now())
    return frames

def getSubjectsListFromFolder(dataFolder):
    allNames = [n for n in os.listdir(dataFolder)]
    allNames = set([n[:2] for n in allNames])
    names = []
    for n in allNames:
        try:
            names.append(int(n[-2:]))
        except ValueError:
            continue
    return sorted(names)

def readBIWI_Frames(dataFolder = BIWI_Data_folder, subjectList = None):
    #print('Frames from ' + str(dataFolder) + ' have been started to read by ' + now())
    biwiFrames = {}
    if subjectList == None: subjectList = getSubjectsListFromFolder(dataFolder)
    for subj in subjectList:
        frames = getAllFramesForSubj(subj, dataFolder)
        biwiFrames[subj] = frames
    return biwiFrames

def showSampleFrames(count = 10):
    biwiFrames = readBIWI_Frames(dataFolder = BIWI_SnippedData_folder)
    for subj, frames in biwiFrames.items():
        frames = [(n, f) for n, f in sorted(frames, key=lambda x: x[0])]
        for name, frame in frames[:count]:
            print(frame.shape)
            pyplot.imshow(numpy.rollaxis(frame, 0, 3))
            pyplot.title(name)
            pyplot.show()
    
#################### Merging ####################
def scaleX(arr):
    return new_arr

def scaleY(arr):
    new_arr = arr/label_rescaling_factor#
    return new_arr

def rolling_window(m, timesteps):
    shape = (m.shape[0] - timesteps + 1, timesteps) + m.shape[1:]
    strides = (m.strides[0],) + m.strides
    return numpy.lib.stride_tricks.as_strided(m, shape=shape, strides=strides)

def reshaper(m, l, timesteps, overlapping):
    if overlapping:
        m= rolling_window(m, timesteps)
        l = l[timesteps-1:]
    else:
        wasted = (m.shape[0] % timesteps)
        m, l = m[wasted:], l[wasted:]
        m = m.reshape((int(m.shape[0]/timesteps), timesteps, m.shape[1], m.shape[2], m.shape[3]))
        l = l.reshape((int(l.shape[0]/timesteps), timesteps, l.shape[1]))
        l = l[:, -1, :]
    return m, l

def labelFramesForSubj(frames, annos, timesteps = None, overlapping = False, scaling = True):
    frames = {n: f for n, f in frames}
    keys = sorted(frames & annos.keys())
    inputMatrix = numpy.stack(itemgetter(*keys)(frames))
    labels = numpy.stack(itemgetter(*keys)(annos))
    if scaling: # scaleX()
        inputMatrix, labels = inputMatrix, scaleY(labels)
    if timesteps != None:
        inputMatrix, labels = reshaper(inputMatrix, labels, timesteps, overlapping)
    return inputMatrix, labels

def readBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, timesteps = None, overlapping = False, scaling = True):
    if subjectList == None: subjectList = [s for s in range(1, 25)]
    biwiFrames = readBIWI_Frames(dataFolder = dataFolder, subjectList = subjectList)
    biwiAnnos = readBIWI_Annos(tarFile = labelsTarFile, subjectList = subjectList)
    labeledFrames = lambda frames, labels: labelFramesForSubj(frames, labels, timesteps, overlapping, scaling)
    biwi = (labeledFrames(frames, biwiAnnos[subj]) for subj, frames in biwiFrames.items())
    print('All frames and annotations from ' + str(len(subjectList)) + ' datasets have been read by ' + now())
    return biwi
   

def printSamplesFromBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None):
    biwi = readBIWIDataset(dataFolder, labelsTarFile, subjectList = subjectList, timesteps = 10, overlapping = True)
    for subj, (inputMatrix, labels) in enumerate(biwi):
        print(subj+1, inputMatrix.shape, labels.shape)

#################### Testing ####################
def main():
    showSampleFrames(1)
    #printSampleAnnos(count = -1)
    #printSampleAnnosForSubj(1, count = -1)
    #printSamplesFromBIWIDataset(subjectList = [1])
   # readBIWIDataset(dataFolder = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
   
if __name__ == "__main__":
    main()
    print('Done')