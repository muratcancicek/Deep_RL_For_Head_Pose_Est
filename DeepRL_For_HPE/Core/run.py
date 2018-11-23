# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

from NeighborFolderimporter import *
from DatasetHandler.BiwiBrowser import *
def main():
    printSamplesFromBIWIDataset(frameTarFile = BIWI_SnippedData_file, labelsTarFile = BIWI_Lebels_file_Local)
    print('This Project will be training a set of Deep Reinforcement Algorithms for Head Pose Estimation.')

if __name__ == "__main__":
    main()
