# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
else:
    from NeighborFolderimporter import *

from DatasetHandler.BiwiBrowser import *
import matplotlib.pyplot as plt

num_datasets = 3
num_outputs = 3
timesteps = None # 1 # 
overlapping = False

def drawSingleDataset(labels):
    output1 = labels[:, :1]
    output2 = labels[:, 1:2]
    output3 = labels[:, 2:3]
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, sharex=True, sharey=True)
    ax1.plot(output1, 'r')
    ax1.set_title('Sharing both axes')
    ax2.plot(output2, 'b')
    ax3.plot(output3, 'g')
    f.subplots_adjust(vspace=0)
    plt.setp([a.get_yticklabels() for a in f.axes[1:]], visible=False)#
    plt.show()

def drawPlots(labelSets, subjectIDs, sets = None):
    colors = ['r', 'b', 'g']
    titles = ['Pitch', 'Yaw', 'Roll']
    red = (1.0, 0.95, 0.95)
    blue = (0.95, 0.95, 1.0)
    if sets == None: sets = [i for i in range(len(labelSets))]
    f, rows = plt.subplots(len(labelSets), 3, sharey=True,  sharex=True,figsize=(21, 4*len(labelSets)))
    for col in range(len(rows[0])):
        rows[0][col].set_title(titles[col])

    row = 0
    for labels in labelSets:
    #for row in range(len(rows)):
        for col in range(len(rows[row])):
            rows[row][col].plot(labels[:, num_outputs+col], colors[col])
            rows[row][col].set_facecolor(red if 'F' in subjectIDs[row] else blue)
            rows[row][0].set_ylabel('Set: %d, Subj: %s, Len: %d' % (sets[row]+1, subjectIDs[row], labels.shape[0]))
        row += 1

    f.subplots_adjust(hspace=0, wspace=0)
    #plt.setp([a.get_yticklabels() for a in f.axes[1:]], visible=False)
    plt.show()

def main():
    #biwi = readBIWIDataset(subjectList = [s for s in range(1, num_datasets+1)], timesteps = timesteps, overlapping = overlapping)#
    biwiAnnos = readBIWI_AnnosAsMatrix(subjectList = [s for s in range(1, num_datasets+1)])
    subjectIDs = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'F03', 'M09', 'M10', 'F05', 'M11', 'M12', 'F02', 'M01', 'M13', 'M14']
    drawPlots(biwiAnnos, subjectIDs)

    
if __name__ == "__main__":
    main()
    print('Done')

    