# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import os

if 'COMPUTERNAME' in os.environ:
    BIWI_Main_Folder = "C:/cStorage/Datasets/hpdb/"
else:
    BIWI_Main_Folder = "/home/mcicek/Datasets/HeadPoses/biwi/"