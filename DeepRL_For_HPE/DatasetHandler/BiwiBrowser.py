# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
import tarfile
from os import listdir
from paths import *

BIWI_Data_file = BIWI_Main_Folder + 'kinect_head_pose_db.tgz'
BIWI_Lebels_file = BIWI_Main_Folder + 'db_annotations.tgz'

tar = tarfile.open(BIWI_Data_file, "r:gz")
for member in tar.getmembers()[:6]:
    print(member)