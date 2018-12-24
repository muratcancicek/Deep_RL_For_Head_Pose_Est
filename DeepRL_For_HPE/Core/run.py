# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *

from DatasetHandler.BiwiBrowser import *

def main():
    print('This Project will be training a set of Deep Reinforcement Algorithms for Head Pose Estimation.')
    printSamplesFromBIWIDataset(dataFolder = BIWI_SnippedData_folder, 
                                labelsTarFile = BIWI_Lebels_file_Local, 
                                subjectList = [1])

if __name__ == "__main__":
    main()
