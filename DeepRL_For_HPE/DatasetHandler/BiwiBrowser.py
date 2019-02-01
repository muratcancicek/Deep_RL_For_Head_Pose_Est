# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import os
if 'COMPUTERNAME' in os.environ:
    if os.environ['COMPUTERNAME'] == "MSI3":
        os.environ['KERAS_BACKEND'] = 'theano'
        os.environ['MKL_THREADING_LAYER'] = 'GNU'

# Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from DatasetHandler.NeighborFolderimporter import *
    from DatasetHandler.BiwiTarBrowser import *
else:
    from NeighborFolderimporter import *
    from BiwiTarBrowser import *

from keras.applications import vgg16
from sklearn.preprocessing import MinMaxScaler, scale
from keras.preprocessing import image
from matplotlib import pyplot
from os import listdir
import itertools
import datetime
import tarfile
import struct
import random
import numpy
import png
import cv2
import os
random.seed(7)
importNeighborFolders()
from paths import *

#################### Constants ####################
pwd = os.path.abspath(os.path.dirname(__file__))
BIWI_Data_folder = BIWI_Main_Folder + 'hpdb/'
BIWI_SnippedData_folder = pwd + '/BIWI_Files/BIWI_Samples/hpdb/'.replace('/', os.path.sep)
BIWI_Lebels_file = BIWI_Main_Folder + 'db_annotations.tgz'
BIWI_Lebels_file_Local = pwd + '/BIWI_Files/db_annotations.tgz'.replace('/', os.path.sep)
BIWI_Frame_Shape = (480, 640, 3)
Target_Frame_Shape_VGG16 = (240, 320, 3)
def now(): return str(datetime.datetime.now())
label_rescaling_factor = 100
BIWI_Subject_IDs = ['XX', 'F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'F03', 'M09', 'M10', 'F05', 'M11', 'M12', 'F02', 'M01', 'M13', 'M14']
BIWI_Lebel_Scalers = getAnnoScalers(tarFile = BIWI_Lebels_file)
#################### Frame Reading ####################
def getRGBpngFileName(subject, frame):
    return str(subject).zfill(2) + '/frame_' + str(frame).zfill(5) + '_rgb.png'

def pngObjToNpArr(imagePath):
    img = image.load_img(imagePath, target_size = Target_Frame_Shape_VGG16)
    x = image.img_to_array(img)
    x = x[8:-8, 48:-48, :]#[14:-15, 74:-75, :]
    return vgg16.preprocess_input(x)

def getBIWIFrameAsNpArr(subject, frame, dataFolder = BIWI_Data_folder, preprocess_input = None):
    imagePath = dataFolder + getRGBpngFileName(subject, frame)
    if preprocess_input == None:
        return pngObjToNpArr(imagePath)
    else:
        return preprocess_input(imagePath)

def filterFrameNamesForSubj(subject, dataFolder):
    subjectFolder = str(subject).zfill(2) + os.path.sep
    allNames = os.listdir(dataFolder + subjectFolder)
    frameNamesForSubj = (fn for fn in allNames if '_rgb.png' in fn)
    frameKey = lambda n: str(subject).zfill(2) + '/' + n[:-8]
    absolutePath = lambda n: dataFolder + subjectFolder + n
    frameNamesForSubj = ((frameKey(n), absolutePath(n)) for n in sorted(frameNamesForSubj))
    return frameNamesForSubj

def getAllFramesForSubj(subject, dataFolder = BIWI_Data_folder, preprocess_input = None):
    frameNamesForSubj = filterFrameNamesForSubj(subject, dataFolder)
    #print('Subject ' + str(subject).zfill(2) + '\'s frames have been started to read ' + now())
    if preprocess_input == None: preprocess_input = pngObjToNpArr
    frames = ((n, preprocess_input(framePath)) for n, framePath in frameNamesForSubj)
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

def readBIWI_Frames(dataFolder = BIWI_Data_folder, subjectList = None, preprocess_input = None):
    #print('Frames from ' + str(dataFolder) + ' have been started to read by ' + now())
    biwiFrames = {}
    if subjectList == None: subjectList = getSubjectsListFromFolder(dataFolder)
    for subj in subjectList:
        frames = getAllFramesForSubj(subj, dataFolder, preprocess_input = preprocess_input)
        biwiFrames[subj] = frames
    return biwiFrames

def showSampleFrames(count = 10, preprocess_input = None):
    biwiFrames = readBIWI_Frames(dataFolder = BIWI_SnippedData_folder, preprocess_input = preprocess_input)
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
    new_arr = arr/label_rescaling_factor#+100
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

def labelFramesForSubj(frames, annos, timesteps = None, overlapping = False, scaling = True, scalers = None):
    frames = {n: f for n, f in frames}
    keys = sorted(frames & annos.keys())
    inputMatrix = numpy.stack(itemgetter(*keys)(frames))
    labels = numpy.stack(itemgetter(*keys)(annos))
    if scaling: # scaleX()
        #inputMatrix, labels = inputMatrix, scaleY(labels)
        if scalers != None: labels = scaleAnnoByScalers(labels, scalers)
    if timesteps != None:
        inputMatrix, labels = reshaper(inputMatrix, labels, timesteps, overlapping)
    return inputMatrix, labels

def readBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, timesteps = None, overlapping = False, scaling = True, preprocess_input = None, printing = True):
    if subjectList == None: subjectList = [s for s in range(1, 25)]
    biwiFrames = readBIWI_Frames(dataFolder = dataFolder, subjectList = subjectList, preprocess_input = preprocess_input)
    biwiAnnos = readBIWI_Annos(tarFile = labelsTarFile, subjectList = subjectList)
    scalers = BIWI_Lebel_Scalers #getAnnoScalers(biwiAnnos, tarFile = labelsTarFile, subjectList = subjectList)
    labeledFrames = lambda frames, labels: labelFramesForSubj(frames, labels, timesteps, overlapping, scaling, scalers)
    biwi = (labeledFrames(frames, biwiAnnos[subj]) for subj, frames in biwiFrames.items())
    if printing: print('All frames and annotations from ' + str(len(subjectList)) + ' datasets have been read by ' + now())
    return biwi   

#################### GeneratorForBIWIDataset ####################
def generatorForBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, timesteps = None, overlapping = False, scaling = True, preprocess_input = None, shuffle = True):
    samples_per_epoch = 0
    if shuffle and subjectList != None: random.shuffle(subjectList)
    biwi = readBIWIDataset(dataFolder, labelsTarFile, subjectList, timesteps, overlapping, scaling, preprocess_input, printing = False)
    gen = itertools.chain()
    for inputMatrix, labels in biwi:
        samples_per_epoch += len(inputMatrix)
        z = list(zip(inputMatrix, labels))
        if shuffle: random.shuffle(z)
        data = ((frame, label) for frame, label in z)
        gen = itertools.chain(gen, data)
    return samples_per_epoch, gen

def genBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, timesteps = None, overlapping = False, scaling = True, preprocess_input = None, shuffle = True):
    samples_per_epoch = 0
    if shuffle and subjectList != None: random.shuffle(subjectList)
    biwi = readBIWIDataset(dataFolder, labelsTarFile, subjectList, timesteps, overlapping, scaling, preprocess_input, printing = False)
    fr = itertools.chain()
    lbl = itertools.chain()
    for inputMatrix, labels in biwi:
        samples_per_epoch += len(inputMatrix)
        z = list(zip(inputMatrix, labels))
        if shuffle: random.shuffle(z)
        f = (frame for frame, label in z)
        l = (label for frame, label in z)
        fr = itertools.chain(fr, f)
        lbl = itertools.chain(lbl, l)
    return samples_per_epoch, fr, lbl


def batchGeneratorForBIWIDataset(batch_size, output_begin, num_outputs, dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, timesteps = None, overlapping = False, scaling = True, preprocess_input = None, shuffle = True):
    samples_count = 0
    while True:
        if samples_count == 0:
            samples_per_epoch, gen = generatorForBIWIDataset(dataFolder, labelsTarFile, subjectList, timesteps, overlapping, scaling, preprocess_input, shuffle)
        c = 0
        frames_batch, labels_batch = numpy.zeros((batch_size, 224, 224, 3)), numpy.zeros((batch_size, num_outputs))
        for frame, label in gen:
            if c < batch_size:
                frames_batch[c], labels_batch[c] = frame, label[output_begin:output_begin+num_outputs]
                c += 1
            else:
                samples_count += batch_size
                if samples_count == samples_per_epoch:
                    samples_count = 0
                yield frames_batch, labels_batch

def getGeneratorsForBIWIDataset(epochs, dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, timesteps = None, overlapping = False, scaling = True, preprocess_input = None, shuffle = True):
    def generate(): return generatorForBIWIDataset(dataFolder, labelsTarFile, subjectList, timesteps, overlapping, scaling, preprocess_input, shuffle)
    biwiGenerators = (generate() for e in range(epochs+1))
    samples_per_epoch, gen = next(biwiGenerators)
    return samples_per_epoch, biwiGenerators

def batchGeneratorFromBIWIGenerators(gens, batch_size, output_begin, num_outputs):
    for samples_per_epoch, g in gens:
        c = 0
        frames_batch, labels_batch = numpy.zeros((batch_size, 224, 224, 3)), numpy.zeros((batch_size, num_outputs))
        for frame, label in g:
                if c < batch_size:
                    frames_batch[c], labels_batch[c] = frame, label[output_begin:output_begin+num_outputs]
                    c += 1
                else:
                    yield frames_batch, labels_batch

def countGeneratorForBIWIDataset():
    gen = generatorForBIWIDataset()
    c, f, l = 0, 0, 0
    for frame, label in gen:
        c, f, l = c+1, frame.shape, label.shape
    print(c, f, l) # 15677 (224, 224, 3) (6,)

def printSamplesFromBIWIDataset(dataFolder = BIWI_Data_folder, labelsTarFile = BIWI_Lebels_file, subjectList = None, preprocess_input = None):
    biwi = readBIWIDataset(dataFolder, labelsTarFile, subjectList = subjectList, timesteps = 10, overlapping = True, preprocess_input = preprocess_input)
    for subj, (inputMatrix, labels) in enumerate(biwi):
        print(subj+1, inputMatrix.shape, labels.shape)

#################### Testing ####################
def main():
    showSampleFrames(1)
    #printSampleAnnos(count = -1)
    #printSampleAnnosForSubj(1, count = -1)
    #printSamplesFromBIWIDataset(subjectList = [1])
   # readBIWIDataset(dataFolder = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
   # countGeneratorForBIWIDataset()
if __name__ == "__main__":
    main()
    print('Done')