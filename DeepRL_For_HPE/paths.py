# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import os

if 'COMPUTERNAME' in os.environ:
    BIWI_Main_Folder = "C:/cStorage/Datasets/hpdb/"
    Keras_Models_Folder = "C:/cStorage/Datasets/Keras_Models/"
else:
    BIWI_Main_Folder = "/home/mcicek/Datasets/HeadPoses/biwi/"
    Keras_Models_Folder = "/home/mcicek/Datasets/Keras_Models/"