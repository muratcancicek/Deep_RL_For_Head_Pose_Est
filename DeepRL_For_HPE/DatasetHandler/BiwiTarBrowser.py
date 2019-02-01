# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
# Dirty importing that allows the main author to switch environments easilyos.path.dirname()
if '.' in __name__: 
    from DatasetHandler.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *

import matplotlib
matplotlib.use('agg')

#from keras.preprocessing.image import img_to_array
from matplotlib import pyplot
from operator import itemgetter
from os import listdir
import datetime
import tarfile
import struct
import numpy
import png
import cv2
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
def now(): return str(datetime.datetime.now())

#################### Frame Reading ####################
def getTarRGBFileName(subject, frame):
    return 'hpdb/' + str(subject).zfill(2) + '/frame_' + str(frame).zfill(5) + '_rgb.png'

def get_np_array_from_tar_object(tar_extractfl):
    '''converts a buffer from a tar file in np.array'''
    return numpy.asarray(bytearray(tar_extractfl.read()), dtype=numpy.uint8)

def pngObjToNpArr(fileObj):
    img = cv2.imdecode(get_np_array_from_tar_object(fileObj), 0)
    return img_to_array(img)

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
    frameNamesForSubj = (fn for fn in allNames if isFrameForSubj(fn, subject))
    return sorted(frameNamesForSubj)

def extractPNG(hpdb, frameName):
    hpdb.extractfile(frameName)
    return pngObjToNpArr(framePath)

def getAllFramesForSubj(subject, hpdb = None, tarFile = BIWI_Data_file):
    if hpdb != None:
        allNames = hpdb.getnames()
        frameNamesForSubj = filterFrameNamesForSubj(subject, allNames)
        #print('Subject ' + str(subject).zfill(2) + '\'s frames have been started to read by ' + now())
        #frames = ((n, extractPNG(hpdb, n)) for n in frameNamesForSubj)
        #print('Subject ' + str(subject).zfill(2) + '\'s all frames have been read by ' + now())
        frames = {}
        #print('Subject ' + str(subject).zfill(2) + '\'s frames have been started to read by ' + now())
        for c, frameName in enumerate(frameNamesForSubj):
            #print('Subject ' + str(subject).zfill(2) + '\'s first ' + str(c).zfill(5) + ' frame have started to be extracted by ' + now())
            file = hpdb.extractfile(frameName)
            #print('Subject ' + str(subject).zfill(2) + '\'s first ' + str(c).zfill(5) + ' frame have started to be parsed by ' + now())
            arr = pngObjToNpArr(file)
            frames[frameName[5:-8]] = arr
        #    if c % 10 == 0 and c > 0:#
        #        print('Subject ' + str(subject).zfill(2) + '\'s first ' + str(c).zfill(5) + ' frames have been read by ' + now())
        #print('Subject ' + str(subject).zfill(2) + '\'s all ' + str(len(frames)) + ' frames have been read by ' + now())
        return frames
    else:
        with tarfile.open(tarFile) as hpdb:
            return getAllFramesForSubj(subject, hpdb, tarFile)
        
def getSubjectsListFromFrameTar(tarFile):
    allNames = tarFile.getnames()
    allNames.remove('hpdb')
    allNames = set([n[:7] for n in allNames])
    names = []
    for n in allNames:
        try:
            names.append(int(n[-2:]))
        except ValueError:
            continue
    return sorted(names)

def readBIWI_Frames(tarFile = BIWI_Data_file, subjectList = None):
    print(str(tarFile) + ' has been started to read by ' + now())
    with tarfile.open(tarFile) as hpdb:
        if subjectList == None: subjectList = getSubjectsListFromFrameTar(hpdb)
        biwiFrames = {}
        for subj in subjectList:
            frames = getAllFramesForSubj(subj, hpdb, tarFile)
            biwiFrames[subj] = frames
        return biwiFrames

def showSampleFrames(count = 10):
    biwiFrames = readBIWI_Frames(tarFile = BIWI_SnippedData_file)
    for subj, frames in biwiFrames:
        frames = [(n, f) for name, f in sorted(frames.items(), key=lambda x: x[0])]
        for name, frame in frames[:count]:
            pyplot.imshow(frame)
            pyplot.title(name)
            pyplot.show()

#################### Annotations Reading ####################
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
        return annos
    else:
        with tarfile.open(tarFile) as annoDB:
            return getAllAnnosForSubj(subject, annoDB, tarFile)

def getSubjectsListFromAnnoTar(tarFile):
    allNames = tarFile.getnames()
    allNames = set([n[:2] for n in allNames])
    return sorted([int(n) for n in allNames])

def readBIWI_Annos(tarFile = BIWI_Lebels_file, subjectList = None):
    #print(str(tarFile) + ' has been started to read by ' + now())
    with tarfile.open(tarFile) as annoDB:
        if subjectList == None: subjectList = getSubjectsListFromAnnoTar(annoDB)
        biwiAnnos = {}
        for subj in subjectList:
            annos = getAllAnnosForSubj(subj, annoDB, tarFile)
            biwiAnnos[subj] = annos
        #print(len(biwiAnnos), 'annotations have been read by ' + now())
        return biwiAnnos
    
def readBIWI_AnnosAsMatrix(tarFile = BIWI_Lebels_file, subjectList = None):
    biwiAnnos = readBIWI_Annos(tarFile, subjectList)
    labelSets = []
    for subj, annos in biwiAnnos.items():
        indices = sorted([i for i in annos])
        labelSets.append(numpy.stack([annos[i] for i in indices]))
    return (l for l in labelSets)

def printSampleAnnosForSubj(subjective, count = 10):
    annos = getAllAnnosForSubj(1, tarFile = BIWI_Lebels_file_Local)
    for name, anno in annos[:count]:
        print(name, anno)

def printSampleAnnos(count = 10):
    biwiAnnos = readBIWI_Annos(tarFile = BIWI_Lebels_file_Local)
    #biwiAnnos = getAllAnnosForSubj(1, tarFile = BIWI_Lebels_file_Local)
    for subj, annos in biwiAnnos.items():
        for name, anno in list(annos.items())[:count]:
            print(anno)

def getMaxMinValuesOfAnnos(biwiAnnos = None, tarFile = BIWI_Lebels_file):
    if biwiAnnos == None: biwiAnnos = readBIWI_Annos(tarFile = tarFile)
    maxs = numpy.array([float('-inf') for i in range(6)])
    mins = numpy.array([float('inf') for i in range(6)])
    for subj, annos in biwiAnnos.items():
        for name, anno in annos.items():
#            Mins: [-92.04399872 -87.70659637 754.18200684  -84.3534317  -66.95036316   -69.62425995]
#            Maxs: [ 231.352005    246.68400574 1297.44995117   53.54709625   76.89344025   63.36795807]
            for i in range(6):
                if anno[i] > maxs[i]:
                    maxs[i] = anno[i]
                if anno[i] < mins[i]:
                    mins[i] = anno[i]
    #print('Mins:', mins)
    #print('Maxs:', maxs)
    return [mins, maxs]

def getAnnoScalers(biwiAnnos = None, tarFile = BIWI_Lebels_file):
    mins, maxs = getMaxMinValuesOfAnnos(biwiAnnos = biwiAnnos, tarFile = tarFile)
    return [mins, maxs]

def scaleAnnoByScalers(labels, scalers):
    return ((labels - scalers[0]) - ((scalers[1]-scalers[0])/2)) / ((scalers[1]-scalers[0])/2)

def unscaleAnnoByScalers(labels, scalers):
    return (labels * (scalers[1]-scalers[0])/2) + ((scalers[1]-scalers[0])/2) + scalers[0]

#################### Merging ####################
def labelFramesForSubj(frames, annos, scalers = None):
    keys = sorted(frames.keys() & annos.keys())
    inputMatrix = numpy.stack(itemgetter(*keys)(frames))
    labels = numpy.stack(itemgetter(*keys)(annos))
    if scalers != None: labels = scaleAnnoByScalers(labels, scalers)
    return inputMatrix, labels

def readBIWIDatasetTar(frameTarFile = BIWI_Data_file, labelsTarFile = BIWI_Lebels_file, subjectList = None):
    if subjectList == None: subjectList = [s for s in range(1, 25)]
    biwiFrames = readBIWI_Frames(tarFile = frameTarFile, subjectList = subjectList)
    biwiAnnos = readBIWI_Annos(tarFile = labelsTarFile, subjectList = subjectList)
    scalers = getAnnoScalers(biwiAnnos, tarFile = tarFile, subjectList = subjectList)
    biwi = (labelFramesForSubj(frames, biwiAnnos[subj], scalers) for subj, frames in biwiFrames.items())
    #biwi = {}
    #for subj, frames in biwiFrames.items():
    #    biwi[subj] = labelFramesForSubj(frames, biwiAnnos[subj])
    return biwi
    
def printSamplesFromBIWIDatasetTar(frameTarFile = BIWI_Data_file, labelsTarFile = BIWI_Lebels_file, subjectList = None):
    biwi = readBIWIDatasetTar(frameTarFile, labelsTarFile, subjectList = subjectList)
    for subj, (inputMatrix, labels) in enumerate(biwi):
        print(subj+1, inputMatrix.shape, labels.shape)

#################### Testing ####################
def main():
    #showSampleFrames(1)
    #printSampleAnnos(count = -1)
    #printSampleAnnosForSubj(1, count = -1)
    getMaxMinValuesOfAnnos()
    #printSamplesFromBIWIDatasetTar(frameTarFile = BIWI_SnippedData_file, 
                                #labelsTarFile = BIWI_Lebels_file_Local, 
                                #subjectList = [1])
   # readBIWIDataset(frameTarFile = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
   
if __name__ == "__main__":
    main()
    print('Done')