import os
import sys
def importNeighborFolders():
    module_path = os.path.abspath(os.path.dirname(__file__)[:-5])
    if module_path not in sys.path:
        sys.path.append(module_path)

importNeighborFolders()