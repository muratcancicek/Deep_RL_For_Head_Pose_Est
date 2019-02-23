from keras.engine.training import *


class ReinforcementModel(Model):
    
    def compile(self, optimizer,
                loss=None,
                metrics=None,
                loss_weights=None,
                sample_weight_mode=None,
                weighted_metrics=None,
                target_tensors=None,
                **kwargs):
        """Configures the model for training.

        # Arguments
            optimizer: String (name of optimizer) or optimizer instance.
                See [optimizers](/optimizers).
            loss: String (name of objective function) or objective function.
                See [losses](/losses).
                If the model has multiple outputs, you can use a different loss
                on each output by passing a dictionary or a list of losses.
                The loss value that will be minimized by the model
                will then be the sum of all individual losses.
            metrics: List of metrics to be evaluated by the model
                during training and testing.
                Typically you will use `metrics=['accuracy']`.
                To specify different metrics for different outputs of a
                multi-output model, you could also pass a dictionary,
                such as `metrics={'output_a': 'accuracy'}`.
            loss_weights: Optional list or dictionary specifying scalar
                coefficients (Python floats) to weight the loss contributions
                of different model outputs.
                The loss value that will be minimized by the model
                will then be the *weighted sum* of all individual losses,
                weighted by the `loss_weights` coefficients.
                If a list, it is expected to have a 1:1 mapping
                to the model's outputs. If a tensor, it is expected to map
                output names (strings) to scalar coefficients.
            sample_weight_mode: If you need to do timestep-wise
                sample weighting (2D weights), set this to `"temporal"`.
                `None` defaults to sample-wise weights (1D).
                If the model has multiple outputs, you can use a different
                `sample_weight_mode` on each output by passing a
                dictionary or a list of modes.
            weighted_metrics: List of metrics to be evaluated and weighted
                by sample_weight or class_weight during training and testing.
            target_tensors: By default, Keras will create placeholders for the
                model's target, which will be fed with the target data during
                training. If instead you would like to use your own
                target tensors (in turn, Keras will not expect external
                Numpy data for these targets at training time), you
                can specify them via the `target_tensors` argument. It can be
                a single tensor (for a single-output model), a list of tensors,
                or a dict mapping output names to target tensors.
            **kwargs: When using the Theano/CNTK backends, these arguments
                are passed into `K.function`.
                When using the TensorFlow backend,
                these arguments are passed into `tf.Session.run`.

        # Raises
            ValueError: In case of invalid arguments for
                `optimizer`, `loss`, `metrics` or `sample_weight_mode`.
        """
        self.optimizer = optimizers.get(optimizer)
        self.loss = loss or []
        self.metrics = metrics or []
        self.loss_weights = loss_weights
        self.sample_weight_mode = sample_weight_mode
        self.weighted_metrics = weighted_metrics

        if not self.built:
            # Model is not compilable because
            # it does not know its number of inputs
            # and outputs, nor their shapes and names.
            # We will compile after the first
            # time the model gets called on training data.
            return
        self._is_compiled = True

        # Prepare loss functions.
        if isinstance(loss, dict):
            for name in loss:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_functions = []
            for name in self.output_names:
                if name not in loss:
                    warnings.warn('Output "' + name +
                                  '" missing from loss dictionary. '
                                  'We assume this was done on purpose, '
                                  'and we will not be expecting '
                                  'any data to be passed to "' + name +
                                  '" during training.', stacklevel=2)
                loss_functions.append(losses.get(loss.get(name)))
        elif isinstance(loss, list):
            if len(loss) != len(self.outputs):
                raise ValueError('When passing a list as loss, '
                                 'it should have one entry per model outputs. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss=' +
                                 str(loss))
            loss_functions = [losses.get(l) for l in loss]
        else:
            loss_function = losses.get(loss)
            loss_functions = [loss_function for _ in range(len(self.outputs))]
        self.loss_functions = loss_functions
        weighted_losses = [
            weighted_masked_objective(fn) for fn in loss_functions]
        skip_target_indices = []
        skip_target_weighing_indices = []
        self._feed_outputs = []
        self._feed_output_names = []
        self._feed_output_shapes = []
        self._feed_loss_fns = []
        for i in range(len(weighted_losses)):
            if weighted_losses[i] is None:
                skip_target_indices.append(i)
                skip_target_weighing_indices.append(i)

        # Prepare output masks.
        masks = self.compute_mask(self.inputs, mask=None)
        if masks is None:
            masks = [None for _ in self.outputs]
        masks = to_list(masks)

        # Prepare loss weights.
        if loss_weights is None:
            loss_weights_list = [1. for _ in range(len(self.outputs))]
        elif isinstance(loss_weights, dict):
            for name in loss_weights:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in loss_weights '
                                     'dictionary: "' + name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            loss_weights_list = []
            for name in self.output_names:
                loss_weights_list.append(loss_weights.get(name, 1.))
        elif isinstance(loss_weights, list):
            if len(loss_weights) != len(self.outputs):
                raise ValueError('When passing a list as loss_weights, '
                                 'it should have one entry per model output. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed loss_weights=' +
                                 str(loss_weights))
            loss_weights_list = loss_weights
        else:
            raise TypeError('Could not interpret loss_weights argument: ' +
                            str(loss_weights) +
                            ' - expected a list of dicts.')

        # Prepare targets of model.
        self.targets = []
        self._feed_targets = []
        if target_tensors is not None:
            if isinstance(target_tensors, list):
                if len(target_tensors) != len(self.outputs):
                    raise ValueError(
                        'When passing a list as `target_tensors`, '
                        'it should have one entry per model output. '
                        'The model has ' + str(len(self.outputs)) +
                        ' outputs, but you passed target_tensors=' +
                        str(target_tensors))
            elif isinstance(target_tensors, dict):
                for name in target_tensors:
                    if name not in self.output_names:
                        raise ValueError('Unknown entry in `target_tensors` '
                                         'dictionary: "' + name + '". '
                                         'Only expected the following keys: ' +
                                         str(self.output_names))
                tmp_target_tensors = []
                for name in self.output_names:
                    tmp_target_tensors.append(target_tensors.get(name, None))
                target_tensors = tmp_target_tensors
            elif K.is_tensor(target_tensors):
                if len(self.outputs) != 1:
                    raise ValueError('The model has ' + str(len(self.outputs)) +
                                     ' outputs, but you passed a single tensor as '
                                     '`target_tensors`. Expected a list or a dict '
                                     'of tensors.')
                target_tensors = [target_tensors]
            else:
                raise TypeError('Expected `target_tensors` to be a tensor, '
                                'a list of tensors, or dict of tensors, but got:',
                                target_tensors)

        for i in range(len(self.outputs)):
            if i in skip_target_indices:
                self.targets.append(None)
            else:
                shape = K.int_shape(self.outputs[i])
                name = self.output_names[i]
                if target_tensors is not None:
                    target = target_tensors[i]
                else:
                    target = None
                if target is None or K.is_placeholder(target):
                    if target is None:
                        target = K.placeholder(
                            ndim=len(shape),
                            name=name + '_target',
                            sparse=K.is_sparse(self.outputs[i]),
                            dtype=K.dtype(self.outputs[i]))
                    self._feed_targets.append(target)
                    self._feed_outputs.append(self.outputs[i])
                    self._feed_output_names.append(name)
                    self._feed_output_shapes.append(shape)
                    self._feed_loss_fns.append(self.loss_functions[i])
                else:
                    skip_target_weighing_indices.append(i)
                self.targets.append(target)

        # Prepare sample weights.
        sample_weights = []
        sample_weight_modes = []
        if isinstance(sample_weight_mode, dict):
            for name in sample_weight_mode:
                if name not in self.output_names:
                    raise ValueError('Unknown entry in '
                                     'sample_weight_mode dictionary: "' +
                                     name + '". '
                                     'Only expected the following keys: ' +
                                     str(self.output_names))
            for i, name in enumerate(self.output_names):
                if i in skip_target_weighing_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    if name not in sample_weight_mode:
                        raise ValueError('Output "' + name +
                                         '" missing from sample_weight_modes '
                                         'dictionary')
                    if sample_weight_mode.get(name) == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        elif isinstance(sample_weight_mode, list):
            if len(sample_weight_mode) != len(self.outputs):
                raise ValueError('When passing a list as sample_weight_mode, '
                                 'it should have one entry per model output. '
                                 'The model has ' + str(len(self.outputs)) +
                                 ' outputs, but you passed '
                                 'sample_weight_mode=' +
                                 str(sample_weight_mode))
            for i in range(len(self.output_names)):
                if i in skip_target_weighing_indices:
                    weight = None
                    sample_weight_modes.append(None)
                else:
                    mode = sample_weight_mode[i]
                    name = self.output_names[i]
                    if mode == 'temporal':
                        weight = K.placeholder(ndim=2,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append('temporal')
                    else:
                        weight = K.placeholder(ndim=1,
                                               name=name + '_sample_weights')
                        sample_weight_modes.append(None)
                sample_weights.append(weight)
        else:
            for i, name in enumerate(self.output_names):
                if i in skip_target_weighing_indices:
                    sample_weight_modes.append(None)
                    sample_weights.append(None)
                else:
                    if sample_weight_mode == 'temporal':
                        sample_weights.append(
                            K.placeholder(ndim=2,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append('temporal')
                    else:
                        sample_weights.append(
                            K.placeholder(ndim=1,
                                          name=name + '_sample_weights'))
                        sample_weight_modes.append(None)
        self.sample_weight_modes = sample_weight_modes
        self._feed_sample_weight_modes = []
        for i in range(len(self.outputs)):
            if i not in skip_target_weighing_indices:
                self._feed_sample_weight_modes.append(
                    self.sample_weight_modes[i])

        # Prepare metrics.
        self.metrics_names = ['loss']
        self.metrics_tensors = []

        # Compute total loss.
        total_loss = None
        with K.name_scope('loss'):
            for i in range(len(self.outputs)):
                if i in skip_target_indices:
                    continue
                y_true = self.targets[i]
                y_pred = self.outputs[i]
                weighted_loss = weighted_losses[i]
                sample_weight = sample_weights[i]
                mask = masks[i]
                loss_weight = loss_weights_list[i]
                with K.name_scope(self.output_names[i] + '_loss'):
                    output_loss = weighted_loss(y_true, y_pred,
                                                sample_weight, mask)
                if len(self.outputs) > 1:
                    self.metrics_tensors.append(output_loss)
                    self.metrics_names.append(self.output_names[i] + '_loss')
                if total_loss is None:
                    total_loss = loss_weight * output_loss
                else:
                    total_loss += loss_weight * output_loss
            if total_loss is None:
                if not self.losses:
                    raise ValueError('The model cannot be compiled '
                                     'because it has no loss to optimize.')
                else:
                    total_loss = 0.

            # Add regularization penalties
            # and other layer-specific losses.
            for loss_tensor in self.losses:
                total_loss += loss_tensor

        # List of same size as output_names.
        # contains tuples (metrics for output, names of metrics).
        nested_metrics = collect_metrics(metrics, self.output_names)
        nested_weighted_metrics = collect_metrics(weighted_metrics,
                                                  self.output_names)
        self.metrics_updates = []
        self.stateful_metric_names = []
        self.stateful_metric_functions = []

        def handle_metrics(metrics, weights=None):
            metric_name_prefix = 'weighted_' if weights is not None else ''

            for metric in metrics:
                if metric in ('accuracy', 'acc', 'crossentropy', 'ce'):
                    # custom handling of accuracy/crossentropy
                    # (because of class mode duality)
                    output_shape = K.int_shape(self.outputs[i])
                    if (output_shape[-1] == 1 or
                       self.loss_functions[i] == losses.binary_crossentropy):
                        # case: binary accuracy/crossentropy
                        if metric in ('accuracy', 'acc'):
                            metric_fn = metrics_module.binary_accuracy
                        elif metric in ('crossentropy', 'ce'):
                            metric_fn = metrics_module.binary_crossentropy
                    elif (self.loss_functions[i] ==
                          losses.sparse_categorical_crossentropy):
                        # case: categorical accuracy/crossentropy
                        # with sparse targets
                        if metric in ('accuracy', 'acc'):
                            metric_fn = metrics_module.sparse_categorical_accuracy
                        elif metric in ('crossentropy', 'ce'):
                            metric_fn = (
                                metrics_module.sparse_categorical_crossentropy)
                    else:
                        # case: categorical accuracy/crossentropy
                        if metric in ('accuracy', 'acc'):
                            metric_fn = metrics_module.categorical_accuracy
                        elif metric in ('crossentropy', 'ce'):
                            metric_fn = metrics_module.categorical_crossentropy
                    if metric in ('accuracy', 'acc'):
                            suffix = 'acc'
                    elif metric in ('crossentropy', 'ce'):
                            suffix = 'ce'
                    weighted_metric_fn = weighted_masked_objective(metric_fn)
                    metric_name = metric_name_prefix + suffix
                else:
                    metric_fn = metrics_module.get(metric)
                    weighted_metric_fn = weighted_masked_objective(metric_fn)
                    # Get metric name as string
                    if hasattr(metric_fn, 'name'):
                        metric_name = metric_fn.name
                    else:
                        metric_name = metric_fn.__name__
                    metric_name = metric_name_prefix + metric_name

                with K.name_scope(metric_name):
                    metric_result = weighted_metric_fn(y_true, y_pred,
                                                       weights=weights,
                                                       mask=masks[i])

                # Append to self.metrics_names, self.metric_tensors,
                # self.stateful_metric_names
                if len(self.output_names) > 1:
                    metric_name = self.output_names[i] + '_' + metric_name
                # Dedupe name
                j = 1
                base_metric_name = metric_name
                while metric_name in self.metrics_names:
                    metric_name = base_metric_name + '_' + str(j)
                    j += 1
                self.metrics_names.append(metric_name)
                self.metrics_tensors.append(metric_result)

                # Keep track of state updates created by
                # stateful metrics (i.e. metrics layers).
                if isinstance(metric_fn, Layer) and metric_fn.stateful:
                    self.stateful_metric_names.append(metric_name)
                    self.stateful_metric_functions.append(metric_fn)
                    self.metrics_updates += metric_fn.updates
        with K.name_scope('metrics'):
            for i in range(len(self.outputs)):
                if i in skip_target_indices:
                    continue

                y_true = self.targets[i]
                y_pred = self.outputs[i]
                weights = sample_weights[i]
                output_metrics = nested_metrics[i]
                output_weighted_metrics = nested_weighted_metrics[i]
                handle_metrics(output_metrics)
                handle_metrics(output_weighted_metrics, weights=weights)

        # Prepare gradient updates and state updates.
        self.total_loss = total_loss
        self.sample_weights = sample_weights
        self._feed_sample_weights = []
        for i in range(len(self.sample_weights)):
            if i not in skip_target_weighing_indices:
                self._feed_sample_weights.append(sample_weights[i])

        # Functions for train, test and predict will
        # be compiled lazily when required.
        # This saves time when the user is not using all functions.
        self._function_kwargs = kwargs

        self.train_function = None
        self.test_function = None
        self.predict_function = None

        # Collected trainable weights, sorted in topological order.
        trainable_weights = self.trainable_weights
        self._collected_trainable_weights = trainable_weights


    def _make_train_function(self):
        if not hasattr(self, 'train_function'):
            raise RuntimeError('You must compile your model before using it.')
        self._check_trainable_weights_consistency()
        if self.train_function is None:
            inputs = (self._feed_inputs +
                      self._feed_targets +
                      self._feed_sample_weights)
            if self._uses_dynamic_learning_phase():
                inputs += [K.learning_phase()]

            with K.name_scope('training'):
                with K.name_scope(self.optimizer.__class__.__name__):
                    print("Costumized _make_train_function method")
                    training_updates = self.optimizer.get_updates(
                        params=self._collected_trainable_weights,
                        loss=self.total_loss)
                updates = (self.updates +
                           training_updates +
                           self.metrics_updates)
                # Gets loss and metrics. Updates weights at each call.
                self.train_function = K.function(
                    inputs,
                    [self.total_loss] + self.metrics_tensors,
                    updates=updates,
                    name='train_function',
                    **self._function_kwargs)
