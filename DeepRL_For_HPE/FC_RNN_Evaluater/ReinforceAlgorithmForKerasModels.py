# Author: Muratcan Cicek, https://users.soe.ucsc.edu/~cicekm/

#Dirty importing that allows the main author to switch environments easily
if '.' in __name__:
    from FC_RNN_Evaluater.EvaluationRecorder import printProgressBar
else:
    from EvaluationRecorder import printProgressBar

from keras.initializers import RandomNormal
from keras import backend as k
from keras import losses
import tensorflow as tf
import numpy as np


def drawSamples(model, episodes, sigma, outputs, seed=None):
    samplesShape = ((episodes,)+outputs.shape)
    distribution = RandomNormal(mean=model.outputs, stddev=sigma, seed=seed)
    return distribution(samplesShape)
    
def getRewardsWithBaselinePerEpisode(samples, targets):
    duplicated_targets = np.repeat(targets[np.newaxis, ...])
    duplicated_targets = tf.convert_to_tensor(duplicated_targets, n, axis=0, name='targets', dtype=tf.float32)
    abs_difference = tf.abs(samples - yy, name='abs_difference')
    allRewards = - tf.reduce_mean(abs_difference, axis = -1) - tf.reduce_mean(abs_difference, axis = -1)
    rewardsPerEpisode = tf.reduce_sum(all, axis = -1, name='rewardsPerEpisode')
    baseline = tf.reduce_mean(allRewards, axis = 0, name='baseline')
    allRewardsWithBaseline = allRewards - baseline
    return tf.reduce_sum(allRewardsWithBaseline, axis=-1, name='rewardsWithBaselinePerEpisode')
    
def getGradientsPerEpisode(samples, targets):
    gradients_per_episode = []
    rewards = getRewardsWithBaselinePerEpisode(samples, targets)
    for i in range(samples.shape[0]):
        loss = losses.mean_squared_error(targets, samples[i])
        gradients = k.gradients(loss, model.trainable_weights)
        rewardedGradients = [g*rewards[i] for g in gradients]
        gradients_per_episode.append(rewardedGradients)
    return gradients_per_episode

def getFinalGradients(samples, targets):
    gradients_per_episode = getGradientsPerEpisode(samples, targets)
    stacked_gradients = []
    for i in range(len(gradients_per_episode[0])):
        stacked_gradients.append(tf.stack([gradients[i] for gradients in gradients_per_episode])) 
    return [tf.reduce_mean(g, axis=0) for g in stacked_gradients]

def getUpdatedModelWithGradients(model, gradients):
    for i in range(len(model.trainable_weights)):
        tf.assign_sub(model.trainable_weights[i], gradients[i])
    return model

def reinforceModelForEpoch(model, data_gen, episodes, sigma, steps_per_epoch, epochCount, verbose=1):
    i = 0
    if verbose == 1: printProgressBar(0, steps_per_epoch, prefix = epochCount, suffix = 'Complete', length = 50)
    for (inputMatrix_batch, inputLabels_batch), outputLabels_batch in data_gen:
        tracker_outputs = model.predict([inputMatrix_batch, inputLabels_batch])
        samples = drawSamples(model, episodes, sigma, tracker_outputs)
        final_gradients = getFinalGradients(samples, outputLabels_batch)
        model = getUpdatedModelWithGradients(model, gradients)
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