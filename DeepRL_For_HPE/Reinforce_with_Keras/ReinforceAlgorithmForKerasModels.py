# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from Core.NeighborFolderimporter import *
    from Reinforce_with_Keras.helpers import printProgressBar
else:
    from NeighborFolderimporter import *
    from helpers import printProgressBar

from keras.initializers import RandomNormal
from keras import backend as K
from keras import losses
import tensorflow as tf
import numpy as np


def drawSamples(model, episodes, sigma, outputs, seed=None):
    samplesShape = outputs.shape
    #samplesShape = ((episodes,)+outputs.shape)
    distribution = RandomNormal(mean=model.outputs, stddev=sigma, seed=seed)
    return distribution(samplesShape)
    
def getRewardsWithBaselinePerEpisode(samples, targets):
    print('#33#', targets, samples)
   # duplicated_targets = np.repeat(targets[np.newaxis, ...], repeats = samples.shape[0], axis=0)
    #duplicated_targets = tf.convert_to_tensor(duplicated_targets, name='targets', dtype=tf.float32)
    duplicated_targets = tf.tile(targets, [samples.shape[0],1,1])
    abs_difference = tf.abs(samples - duplicated_targets, name='abs_difference')
    allRewards = - tf.reduce_mean(abs_difference, axis = -1) - tf.reduce_mean(abs_difference, axis = -1)
    rewardsPerEpisode = tf.reduce_sum(allRewards, axis = -1, name='rewardsPerEpisode')
    baseline = tf.reduce_mean(allRewards, axis = 0, name='baseline')
    allRewardsWithBaseline = allRewards - baseline
    return tf.reduce_sum(allRewardsWithBaseline, axis=-1, name='rewardsWithBaselinePerEpisode')
    
def getGradientsPerEpisode(model, samples, targets):
    gradients_per_episode = []
    rewards = getRewardsWithBaselinePerEpisode(samples, targets)
    for i in range(samples.shape[0]):
        loss = losses.mean_absolute_error(targets, samples[i])
        gradients = K.gradients(loss, model.trainable_weights)
        for g in gradients:
            with tf.Session() as sess:
                print(g, rewards[i].eval())
            rewardedGradients.append(tf.multipy(g, rewards[i][0]))
        gradients_per_episode.append(rewardedGradients)
    return gradients_per_episode

def getFinalGradients(model, samples, targets):
    gradients_per_episode = getGradientsPerEpisode(model, samples, targets)
    stacked_gradients = []
    for i in range(len(gradients_per_episode[0])):
        stacked_gradients.append(tf.stack([gradients[i] for gradients in gradients_per_episode])) 
    return [tf.reduce_mean(g, axis=0) for g in stacked_gradients]

def getUpdatedModelWithGradients(model, gradients):
    for i in range(len(model.trainable_weights)):
        tf.assign_sub(model.trainable_weights[i], gradients[i])
    return model

def getOutputTensor(model, inputMatrix_batch, inputLabels_batch):
    im = tf.convert_to_tensor(inputMatrix_batch, name='im', dtype=tf.float32)
    il = tf.convert_to_tensor(inputLabels_batch, name='il', dtype=tf.float32)
    return model([im, il])

def reinforceModelForEpoch(model, data_gen, episodes, sigma, steps_per_epoch, epochCount, verbose=1):
    i = 0
    if verbose == 1: printProgressBar(0, steps_per_epoch, prefix = epochCount, suffix = 'Complete', length = 50)
    for (inputMatrix_batch, inputLabels_batch), outputLabels_batch in data_gen:
        tracker_outputs = model.predict([inputMatrix_batch, inputLabels_batch])
      #  tracker_outputs = getOutputTensor(model, inputMatrix_batch, inputLabels_batch)
        samples = drawSamples(model, episodes, sigma, outputLabels_batch)
        final_gradients = getFinalGradients(model, samples, outputLabels_batch)
        model = getUpdatedModelWithGradients(model, final_gradients)
        i += 1
        if verbose == 1: printProgressBar(i, steps_per_epoch, prefix = epochCount, suffix = 'Complete', length = 50)
    return model

def reinforceModel(model, data_gen, episodes, sigma, steps_per_epoch, epochs, verbose=1):
    for e in range(epochs):
        epochCount = 'Epoch %d/%d:' % (e+1, epochs)
        model = reinforceModelForEpoch(model, data_gen, episodes, sigma, steps_per_epoch, epochCount, verbose)
    return model

def main():
    pass

if __name__ == "__main__":
    main()
    print('Done')