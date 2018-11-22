# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import tarfile
from os import listdir
import numpy
import png
from paths import *

BIWI_Data_file = BIWI_Main_Folder + 'kinect_head_pose_db.tgz'
BIWI_Lebels_file = BIWI_Main_Folder + 'db_annotations.tgz'

def getTarRGBFileName(subject, frame):
    return 'hpdb/' + str(subject).zfill(2) + '/frame_' + str(frame).zfill(5) + '_rgb.png'

with tarfile.open(BIWI_Data_file) as hpdb:

    
    #file = hpdb.extractfile('hpdb/01/frame_00166_rgb.png')

    file = hpdb.extractfile(getTarRGBFileName(1, 166))
    for f in range(166, 176):
        name = getTarRGBFileName(1, f)
        fileName = name.split('/')[-1]
        hpdb.extract(name, path = '/home/mcicek/Projects/deep_rl_for_head_pose_est/DeepRL_For_HPE/DatasetHandler/BIWI_Samples/')
        print(fileName)
    #r=png.Reader(file=file).asFloat() 
    #print(r)
    #image_2d = numpy.vstack(list(map(numpy.uint16,r)))
    #age_3d = numpy.reshape(image_2d, (640, 480, 3))
    #print(aage_3d.size)

print('Done')