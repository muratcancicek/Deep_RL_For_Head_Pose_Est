# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

import numpy
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *

from DatasetHandler.BiwiBrowser import readBIWIDataset, BIWI_Subject_IDs, now, label_rescaling_factor, BIWI_Lebel_Scalers, unscaleAnnoByScalers

trainingBiwi = readBIWIDataset(subjectList = [1])


