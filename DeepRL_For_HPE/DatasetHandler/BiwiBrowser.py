# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
# Dirty importing that allows the main author to switch environments easily
""""""
if 'D:' in os.path.dirname(__file__):
    from NeighborFolderimporter import *
    from BiwiTarBrowser import *
else:
    from .NeighborFolderimporter import *
    from .BiwiTarBrowser import *
from keras.preprocessing.image import img_to_array
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
BIWI_Frame_Shape = (480, 640, 3)
def now(): return str(datetime.datetime.now())

#################### Frame Reading ####################
def getRGBpngFileName(subject, frame):
    return str(subject).zfill(2) + '/frame_' + str(frame).zfill(5) + '_rgb.png'

def pngObjToNpArr(imagePath):
    image = cv2.imread(imagePath)
    return img_to_array(image)

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
    print('Subject ' + str(subject).zfill(2) + '\'s frames have been started to read ' + now())
    frames = ((n, pngObjToNpArr(framePath)) for n, framePath in frameNamesForSubj)
    print('Subject ' + str(subject).zfill(2) + '\'s all frames have been read by ' + now())
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
    print('Frames from ' + str(dataFolder) + ' have been started to read by ' + now())
    biwiFrames = {}
    if subjectList == None: subjectList = getSubjectsListFromFolder(dataFolder)
    for subj in subjectList:
        frames = getAllFramesForSubj(subj, dataFolder)
        biwiFrames[subj] = frames
    return biwiFrames

def showSampleFrames(count = 10):
    biwiFrames = readBIWI_Frames(dataFolder = BIWI_SnippedData_folder)
    for subj, frames in biwiFrames.items():
        frames = [(n, f) for n, f in sorted(frames.items(), key=lambda x: x[0])]
        for name, frame in frames[:count]:
            pyplot.imshow(frame)
            pyplot.title(name)
            pyplot.show()
    
#################### Merging ####################
def labelFramesForSubj(frames, annos):
    frames = {n: f for n, f in frames}
    keys = sorted(frames & annos.keys())
    inputMatrix = numpy.stack(itemgetter(*keys)(frames))
    labels = numpy.stack(itemgetter(*keys)(annos))
    return inputMatrix, labels

def readBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None):
    if subjectList == None: subjectList = [s for s in range(1, 25)]
    biwiFrames = readBIWI_Frames(dataFolder = dataFolder, subjectList = subjectList)
    biwiAnnos = readBIWI_Annos(tarFile = labelsTarFile, subjectList = subjectList)
    biwi = (labelFramesForSubj(frames, biwiAnnos[subj]) for subj, frames in biwiFrames.items())
    #for subj, frames in biwiFrames.items():
    #    biwi[subj] = labelFramesForSubj(frames, biwiAnnos[subj])
    #    print('Frames for ' + str(subj) + ' have been labeled by ' + now())
    return biwi
    
def printSamplesFromBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None):
    biwi = readBIWIDataset(dataFolder, labelsTarFile, subjectList = subjectList)
    for subj, (inputMatrix, labels) in biwi.items():
        print(subj, inputMatrix.shape, labels.shape)

#################### Testing ####################
def main():
    #showSampleFrames(1)
    #printSampleAnnos(count = -1)
    #printSampleAnnosForSubj(1, count = -1)
    printSamplesFromBIWIDataset(dataFolder = BIWI_SnippedData_folder, 
                                labelsTarFile = BIWI_Lebels_file_Local, 
                                subjectList = [1])
   # readBIWIDataset(dataFolder = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
   
if __name__ == "__main__":
    main()
    print('Done')