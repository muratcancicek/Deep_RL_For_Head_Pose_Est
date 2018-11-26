import os
import sys
def importNeighborFolders():
    lenOfcurrentFolderName = len(os.path.dirname(__file__).split(os.path.sep)[-1])+1
    module_path = os.path.abspath(os.path.dirname(__file__)[:-lenOfcurrentFolderName])
    if module_path not in sys.path:
        sys.path.append(module_path)

importNeighborFolders()