# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/
#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from FC_RNN_Evaluater.EvaluationRecorder import *
else:
    from NeighborFolderimporter import *
    from EvaluationRecorder import *

importNeighborFolders()
from DatasetHandler.BiwiBrowser import BIWI_Subject_IDs, label_rescaling_factor, BIWI_Lebel_Scalers
import matplotlib.pyplot as plt

def drawPlotsForSubj(outputs, subj, subjID, modelID, num_outputs, angles):
    if num_outputs == 1: 
        angles = ['Yaw']
        BIWI_Lebel_Scalers[0] = [BIWI_Lebel_Scalers[0][1]]
        BIWI_Lebel_Scalers[1] = [BIWI_Lebel_Scalers[1][1]]
    colors = ['#FFAA00', '#00AA00', '#0000AA', '#AA0000'] 
    title = 'Estimations for the Subject %d (Subject ID: %s, Total length: %d)\nby the Model %s' % (subj, subjID, outputs[0][0].shape[0], modelID)
    red, blue = (1.0, 0.95, 0.95), (0.95, 0.95, 1.0)
    f, rows = plt.subplots(num_outputs, 1, sharex=True, figsize=(16, 3*num_outputs))
    f.suptitle(title)
    for i, (matrix, absolute_mean_error) in enumerate(outputs):
        cell = rows
        if num_outputs > 1: cell = rows[i]
        l1 = cell.plot(matrix[:, 0], colors[i], label='Ground-truth')
        l2 = cell.plot(matrix[:, 1], colors[-1], label='Estimation')
        cell.set_facecolor(red if 'F' in subjID else blue)
        #cell.set_xlim([0, 1000]), sharey=True
        #print([BIWI_Lebel_Scalers[0][3+i]-10, BIWI_Lebel_Scalers[1][3+i]+10])
        cell.set_ylim([BIWI_Lebel_Scalers[0][3+i]-10, BIWI_Lebel_Scalers[1][3+i]+10])
        cell.set_ylabel('%s Angle\nAbsolute Mean Error: %.2f' % (angles[i], absolute_mean_error))
    f.subplots_adjust(top=0.93, hspace=0, wspace=0)
    return f

def drawResults(results, modelStr, modelID, num_outputs, angles, save = False):
    figures = []
    for subject, outputs in results:
        f = drawPlotsForSubj(outputs, subject, BIWI_Subject_IDs[subject], modelStr, num_outputs, angles = angles)
        figures.append((f, subject))
    if save:
        for f, subj in figures:
            fileName = 'subject%d_%s.png' % (subj, modelID)
            f.savefig(addModelFolder(CURRENT_MODEL, fileName), bbox_inches='tight')
            printLog(fileName, 'has been saved by %s.' % now(), record = save)
    return figures
