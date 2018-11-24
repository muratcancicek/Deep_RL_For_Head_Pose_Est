# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
# Dirty importing that allows the main author to switch environments easily
if len(os.path.dirname(__file__)) == 0 or 'D:' in os.path.dirname(__file__):
    from NeighborFolderimporter import *
else:
    from DatasetHandler.NeighborFolderimporter import *
from matplotlib import pyplot
from os import listdir
import tarfile
import struct
import numpy
import png
import os

importNeighborFolders()
from paths import *

#################### Constants ####################
pwd = os.path.abspath(os.path.dirname(__file__))
BIWI_Data_file = BIWI_Main_Folder + 'kinect_head_pose_db.tgz'
BIWI_SnippedData_file = pwd + '/BIWI_Files/BIWI_Samples/SnippedBiwi.tgz'.replace('/', os.path.sep)
BIWI_Lebels_file = BIWI_Main_Folder + 'db_annotations.tgz'
BIWI_Lebels_file_Local = pwd + '/BIWI_Files/db_annotations.tgz'.replace('/', os.path.sep)
BIWI_Frame_Shape = (480, 640, 3)

#################### Frame Reading ####################
def getTarRGBFileName(subject, frame):
    return 'hpdb/' + str(subject).zfill(2) + '/frame_' + str(frame).zfill(5) + '_rgb.png'

def pngObjToNpArr(fileObj):
    pngDict = png.Reader(file=fileObj).asFloat() 
    arr = numpy.zeros(BIWI_Frame_Shape)
    for i, r in enumerate(pngDict[2]):
        row = numpy.fromiter(r, dtype=numpy.float)
        arr[i] = numpy.reshape(row, (BIWI_Frame_Shape[1], BIWI_Frame_Shape[2]))
    return arr

def getBIWIFrameAsNpArr(subject, frame, hpdb = None, tarFile = BIWI_Data_file):
    if hpdb != None:
        file = hpdb.extractfile(getTarRGBFileName(subject, frame))
        return pngObjToNpArr(file)
    else:
        with tarfile.open(tarFile) as hpdb:
            return getBIWIFrameAsNpArr(subject, frame, hpdb, tarFile)

def isFrameForSubj(fileName, subject):
    isFrame = '_rgb.png' in fileName
    isForSubj =  ('hpdb/' + str(subject).zfill(2) + '/') in fileName
    return isFrame and isForSubj

def filterFrameNamesForSubj(subject, allNames):
    frameNamesForSubj = filter(lambda fn: isFrameForSubj(fn, subject), allNames)
    return sorted(frameNamesForSubj)

def getAllFramesForSubj(subject, hpdb = None, tarFile = BIWI_Data_file):
    if hpdb != None:
        allNames = hpdb.getnames()
        frameNamesForSubj = filterFrameNamesForSubj(subject, allNames)
        frames = {}
        for frameName in frameNamesForSubj:
            file = hpdb.extractfile(frameName)
            arr = pngObjToNpArr(file)
            frames[frameName[5:-8]] = arr
        print('Subject ' + str(subject).zfill(2) + '\'s frames have been read.')
        return frames
    else:
        with tarfile.open(tarFile) as hpdb:
            return getAllFramesForSubj(subject, hpdb, tarFile)
        
def getSubjectsListFromFrameTar(tarFile):
    allNames = tarFile.getnames()
    allNames.remove('hpdb')
    allNames = set([n[:7] for n in allNames])
    return [int(n[-2:]) for n in allNames]

def readBIWI_Frames(tarFile = BIWI_Data_file):
    with tarfile.open(tarFile) as hpdb:
        subjects = getSubjectsListFromFrameTar(hpdb)
        biwiFrames = {}
        for subj in subjects:
            frames = getAllFramesForSubj(subj, hpdb, tarFile)
            biwiFrames[subj] = frames
        return biwiFrames

def showSampleFrames(count = 10):
    biwiFrames = readBIWI_Frames(tarFile = BIWI_SnippedData_file)
    for subj, frames in biwiFrames:
        for name, frame in frames[:count]:
            pyplot.imshow(frame)
            pyplot.title(name)
            pyplot.show()

#################### Annotations Reading ####################import struct
def parseAnno(file):
    floats = struct.unpack('ffffff', file.read(24))
    return numpy.array(floats, dtype = float)

def isAnnoForSubj(fileName, subject):
    isFrame = '_pose.bin' in fileName
    isForSubj =  (str(subject).zfill(2) + '/') in fileName
    return isFrame and isForSubj

def filterAnnoNamesForSubj(subject, allNames):
        annoNamesForSubj = filter(lambda fn: isAnnoForSubj(fn, subject), allNames)
        return sorted(annoNamesForSubj)

def getAllAnnosForSubj(subject, annoDB = None, tarFile = BIWI_Data_file):
    if annoDB != None:
        allNames = annoDB.getnames()
        annoNamesForSubj = filterAnnoNamesForSubj(subject, allNames)
        annos = {}
        for annoName in annoNamesForSubj:
            file = annoDB.extractfile(annoName)
            anno = parseAnno(file)
            annos[annoName[:-9]] = anno
        print('Subject ' + str(subject).zfill(2) + '\'s annotations have been read.')
        return annos
    else:
        with tarfile.open(tarFile) as annoDB:
            return getAllAnnosForSubj(subject, annoDB, tarFile)

def getSubjectsListFromAnnoTar(tarFile):
    allNames = tarFile.getnames()
    allNames = set([n[:2] for n in allNames])
    return sorted([int(n) for n in allNames])

def readBIWI_Annos(tarFile = BIWI_Lebels_file):
    with tarfile.open(tarFile) as annoDB:
        subjects = getSubjectsListFromAnnoTar(annoDB)
        biwiAnnos = {}
        for subj in subjects:
            annos = getAllAnnosForSubj(subj, annoDB, tarFile)
            biwiAnnos[subj] = annos
        return biwiAnnos

def printSampleAnnosForSubj(subjective, count = 10):
    annos = getAllAnnosForSubj(1, tarFile = BIWI_Lebels_file_Local)
    for name, anno in annos[:count]:
        print(name, anno)

def printSampleAnnos(count = 10):
    biwiAnnos = readBIWI_Annos(tarFile = BIWI_Lebels_file_Local)
    biwiAnnos = getAllAnnosForSubj(1, tarFile = BIWI_Lebels_file_Local)
    for subj, annos in biwiAnnos:
        for name, anno in annos[:count]:
            print(anno)
    
#################### Merging ####################

def labelFramesForSubj(frames, annos):
    labeledData = []
    for frameName, frame in frames.items():
        if frameName in annos.keys():
            labeledData.append((frame, annos[frameName]))
    if len(labeledData) == 0:
        print('Subject has no data')
        return None, None
    inputMatrix = numpy.zeros((len(labeledData), BIWI_Frame_Shape[0], 
                               BIWI_Frame_Shape[1], BIWI_Frame_Shape[2]))
    labels = numpy.zeros((len(labeledData), len(labeledData[0][1])))
    for i, (frame, anno) in enumerate(labeledData):
        inputMatrix[i] = frame
        labels[i] = anno
    return inputMatrix, labels

def readBIWIDataset(frameTarFile = BIWI_Data_file, labelsTarFile = BIWI_Lebels_file):
    biwiFrames = readBIWI_Frames(tarFile = frameTarFile)
    biwiAnnos = readBIWI_Annos(tarFile = labelsTarFile)
    biwi = {}
    for subj, frames in biwiFrames.items():
        biwi[subj] = labelFramesForSubj(frames, biwiAnnos[subj])
    return biwi
    
def printSamplesFromBIWIDataset(frameTarFile = BIWI_Data_file, labelsTarFile = BIWI_Lebels_file):
    biwi = readBIWIDataset(frameTarFile, labelsTarFile)
    for subj, (inputMatrix, labels) in biwi.items():
        print(subj, inputMatrix.shape, labels.shape)


#################### Testing ####################
def main():
    #showSampleFrames(1)
    #printSampleAnnos(count = -1)
    #printSampleAnnosForSubj(1, count = -1)
    printSamplesFromBIWIDataset(frameTarFile = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
   # readBIWIDataset(frameTarFile = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
   
if __name__ == "__main__":
    main()
    print('Done')