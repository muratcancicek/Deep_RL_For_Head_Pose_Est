from keras.optimizers import *
from ReinforceAlgorithmForKerasModels import *

class AdamForRL(Adam):
    

    def _drawSamples(self, outputs, episodes = 5, sigma = 0.01, seed=None):
        samplesShape = outputs.shape
        #samplesShape = ((episodes,)+outputs.shape)
        distribution = RandomNormal(outputs, stddev=sigma, seed=seed)
        return distribution(samplesShape)
    
def _getGradientsPerEpisode(samples, targets, params):
    gradients_per_episode = []
    rewards = getRewardsWithBaselinePerEpisode(samples, targets)
    for i in range(samples.shape[0]):
        loss = losses.mean_absolute_error(targets, samples[i])
        gradients = K.gradients(loss, params)
        rewardedGradients = [g*rewards[i] for g in gradients]
        gradients_per_episode.append(rewardedGradients)
    return gradients_per_episode

    def _getFinalGradients(self, outputs, targets, params):
        samples = self._drawSamples(outputs)
        gradients_per_episode = self._getGradientsPerEpisode(samples, targets, params)
        stacked_gradients = []
        for i in range(len(gradients_per_episode[0])):
            stacked_gradients.append(K.stack([gradients[i] for gradients in gradients_per_episode])) 
        return [K.reduce_mean(g, axis=0) for g in stacked_gradients]
    
    def get_gradients(self, targets, outputs, loss, params):

        print("Costumized AdamForRL(Adam) get_gradients method")
        #grads = K.gradients(loss, params)
        grads = self._getFinalGradients(outputs, targets, params)
        if None in grads:
            raise ValueError('An operation has `None` for gradient. '
                             'Please make sure that all of your ops have a '
                             'gradient defined (i.e. are differentiable). '
                             'Common ops without gradient: '
                             'K.argmax, K.round, K.eval.')
        if hasattr(self, 'clipnorm') and self.clipnorm > 0:
            norm = K.sqrt(sum([K.sum(K.square(g)) for g in grads]))
            grads = [clip_norm(g, self.clipnorm, norm) for g in grads]
        if hasattr(self, 'clipvalue') and self.clipvalue > 0:
            grads = [K.clip(g, -self.clipvalue, self.clipvalue) for g in grads]
        return grads
    
    @interfaces.legacy_get_updates_support
    def get_updates(self, targets, outputs, loss, params):

        grads = self.get_gradients(loss, targets, outputs, params)
        self.updates = [K.update_add(self.iterations, 1)]

        lr = self.lr
        if self.initial_decay > 0:
            lr = lr * (1. / (1. + self.decay * K.cast(self.iterations,
                                                      K.dtype(self.decay))))

        t = K.cast(self.iterations, K.floatx()) + 1
        lr_t = lr * (K.sqrt(1. - K.pow(self.beta_2, t)) /
                     (1. - K.pow(self.beta_1, t)))

        ms = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        vs = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        if self.amsgrad:
            vhats = [K.zeros(K.int_shape(p), dtype=K.dtype(p)) for p in params]
        else:
            vhats = [K.zeros(1) for _ in params]
        self.weights = [self.iterations] + ms + vs + vhats

        for p, g, m, v, vhat in zip(params, grads, ms, vs, vhats):
            m_t = (self.beta_1 * m) + (1. - self.beta_1) * g
            v_t = (self.beta_2 * v) + (1. - self.beta_2) * K.square(g)
            if self.amsgrad:
                vhat_t = K.maximum(vhat, v_t)
                p_t = p - lr_t * m_t / (K.sqrt(vhat_t) + self.epsilon)
                self.updates.append(K.update(vhat, vhat_t))
            else:
                p_t = p - lr_t * m_t / (K.sqrt(v_t) + self.epsilon)

            self.updates.append(K.update(m, m_t))
            self.updates.append(K.update(v, v_t))
            new_p = p_t

            # Apply constraints.
            if getattr(p, 'constraint', None) is not None:
                new_p = p.constraint(new_p)

            self.updates.append(K.update(p, new_p))
        return self.updates
