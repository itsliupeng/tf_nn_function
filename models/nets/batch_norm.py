# -*- coding: utf-8 -*-
# File: batch_norm.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.eager import context
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.training import moving_averages


# decay: being too close to 1 leads to slow start-up. torch use 0.9.
# eps: torch: 1e-5. Lasagne: 1e-4


def get_bn_variables(n_out, use_scale, use_bias, beta_init, gamma_init):
    if use_bias:
        beta = tf.get_variable('syncbn_beta', [n_out], initializer=beta_init)
    else:
        beta = tf.zeros([n_out], name='syncbn_beta')
    if use_scale:
        gamma = tf.get_variable('syncbn_gamma', [n_out], initializer=gamma_init)
    else:
        gamma = tf.ones([n_out], name='syncbn_gamma')
    # x * gamma + beta

    moving_mean = tf.get_variable('syncbn_mean/EMA', [n_out],
                                  initializer=tf.constant_initializer(), trainable=False)
    moving_var = tf.get_variable('syncbn_variance/EMA', [n_out],
                                 initializer=tf.constant_initializer(1.0), trainable=False)
    return beta, gamma, moving_mean, moving_var


def update_bn_ema(xn, batch_mean, batch_var,
                  moving_mean, moving_var, decay, internal_update):
    update_op1 = moving_averages.assign_moving_average(
        moving_mean, batch_mean, decay, zero_debias=False,
        name='syncbn_mean_ema_op')
    update_op2 = moving_averages.assign_moving_average(
        moving_var, batch_var, decay, zero_debias=False,
        name='syncbn_var_ema_op')

    if internal_update:
        with tf.control_dependencies([update_op1, update_op2]):
            return tf.identity(xn, name='syncbn_output')
    else:
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op1)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op2)
        return tf.identity(xn, name='syncbn_output')


class SyncBatchNorm(tf.keras.layers.BatchNormalization):
    def __init__(self,
                 axis=-1,
                 momentum=0.99,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 moving_mean_initializer='zeros',
                 moving_variance_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 renorm=False,
                 renorm_clipping=None,
                 renorm_momentum=0.99,
                 fused=None,
                 trainable=True,
                 virtual_batch_size=None,
                 adjustment=None,
                 name=None,
                 **kwargs):
        super(SyncBatchNorm, self).__init__(
            axis=axis,
            momentum=momentum,
            epsilon=epsilon,
            center=center,
            scale=scale,
            beta_initializer=beta_initializer,
            gamma_initializer=gamma_initializer,
            moving_mean_initializer=moving_mean_initializer,
            moving_variance_initializer=moving_variance_initializer,
            beta_regularizer=beta_regularizer,
            gamma_regularizer=gamma_regularizer,
            beta_constraint=beta_constraint,
            gamma_constraint=gamma_constraint,
            renorm=renorm,
            renorm_clipping=renorm_clipping,
            renorm_momentum=renorm_momentum,
            fused=fused,
            trainable=trainable,
            virtual_batch_size=virtual_batch_size,
            adjustment=adjustment,
            name=name,
            **kwargs)

    def call(self, inputs, training=None, sync=True):
        original_training_value = training
        if training is None:
            training = K.learning_phase()

        in_eager_mode = context.executing_eagerly()
        if self.virtual_batch_size is not None:
            # Virtual batches (aka ghost batches) can be simulated by reshaping the
            # Tensor and reusing the existing batch norm implementation
            original_shape = [-1] + inputs.shape.as_list()[1:]
            expanded_shape = [self.virtual_batch_size, -1] + original_shape[1:]

            # Will cause errors if virtual_batch_size does not divide the batch size
            inputs = array_ops.reshape(inputs, expanded_shape)

            def undo_virtual_batching(outputs):
                outputs = array_ops.reshape(outputs, original_shape)
                return outputs

        if self.fused:
            outputs = self._fused_batch_norm(inputs, training=training)
            if self.virtual_batch_size is not None:
                # Currently never reaches here since fused_batch_norm does not support
                # virtual batching
                outputs = undo_virtual_batching(outputs)
            if not context.executing_eagerly() and original_training_value is None:
                outputs._uses_learning_phase = True  # pylint: disable=protected-access
            return outputs

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.get_shape()
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]
        if self.virtual_batch_size is not None:
            del reduction_axes[1]  # Do not reduce along virtual batch dim

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and
                    len(v.get_shape()) != ndims and
                    reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(self.gamma), _broadcast(self.beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if training_value is not False:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
                # Adjust only during training.
                adj_scale = tf_utils.smart_cond(training,
                                                lambda: adj_scale,
                                                lambda: array_ops.ones_like(adj_scale))
                adj_bias = tf_utils.smart_cond(training,
                                               lambda: adj_bias,
                                               lambda: array_ops.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

            # Some of the computations here are not necessary when training==False
            # but not a constant. However, this makes the code simpler.
            keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
            mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

            moving_mean = self.moving_mean
            moving_variance = self.moving_variance

            mean = tf_utils.smart_cond(training,
                                       lambda: mean,
                                       lambda: moving_mean)
            variance = tf_utils.smart_cond(training,
                                           lambda: variance,
                                           lambda: moving_variance)

            if self.virtual_batch_size is not None:
                # This isn't strictly correct since in ghost batch norm, you are
                # supposed to sequentially update the moving_mean and moving_variance
                # with each sub-batch. However, since the moving statistics are only
                # used during evaluation, it is more efficient to just update in one
                # step and should not make a significant difference in the result.
                new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
                new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
            else:
                new_mean, new_variance = mean, variance

            if self.renorm:
                r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                    new_mean, new_variance, training)
                # When training, the normalized values (say, x) will be transformed as
                # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
                # = x * (r * gamma) + (d * gamma + beta) with renorm.
                r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
                d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
                scale, offset = _compose_transforms(r, d, scale, offset)

            def _do_update(var, value):
                if in_eager_mode and not self.trainable:
                    return

                return self._assign_moving_average(var, value, self.momentum)

            mean_update = tf_utils.smart_cond(
                training,
                lambda: _do_update(self.moving_mean, new_mean),
                lambda: self.moving_mean)
            variance_update = tf_utils.smart_cond(
                training,
                lambda: _do_update(self.moving_variance, new_variance),
                lambda: self.moving_variance)
            if not context.executing_eagerly():
                self.add_update(mean_update, inputs=True)
                self.add_update(variance_update, inputs=True)

        else:
            mean, variance = self.moving_mean, self.moving_variance

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)

        if sync:
            import horovod.tensorflow as hvd
            mean = hvd.allreduce(mean)

        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        if not context.executing_eagerly() and original_training_value is None:
            outputs._uses_learning_phase = True  # pylint: disable=protected-access
        return outputs

# def BatchNorm(inputs, axis=None, training=None, momentum=0.9, epsilon=1e-5,
#               center=True, scale=True,
#               beta_initializer=tf.zeros_initializer(),
#               gamma_initializer=tf.ones_initializer(),
#               virtual_batch_size=None,
#               data_format='channels_last',
#               internal_update=False,
#               sync_statistics=None):
#     """
#     Almost equivalent to `tf.layers.batch_normalization`, but different (and more powerful)
#     in the following:
#
#     1. Accepts an alternative `data_format` option when `axis` is None. For 2D input, this argument will be ignored.
#     2. Default value for `momentum` and `epsilon` is different.
#     3. Default value for `training` is automatically obtained from tensorpack's `TowerContext`, but can be overwritten.
#     4. Support the `internal_update` option, which enables the use of BatchNorm layer inside conditionals.
#     5. Support the `sync_statistics` option, which is very useful in small-batch models.
#
#     Args:
#         internal_update (bool): if False, add EMA update ops to
#           `tf.GraphKeys.UPDATE_OPS`. If True, update EMA inside the layer by control dependencies.
#           They are very similar in speed, but `internal_update=True` can be used
#           when you have conditionals in your model, or when you have multiple networks to train.
#           Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/14699
#         sync_statistics (str or None): one of None, "nccl", or "horovod_estimator".
#
#           By default (None), it uses statistics of the input tensor to normalize.
#           This is the standard way BatchNorm was done in most frameworks.
#
#           When set to "nccl", this layer must be used under tensorpack's multi-GPU trainers.
#           It uses the aggregated statistics of the whole batch (across all GPUs) to normalize.
#
#           When set to "horovod_estimator", this layer must be used under tensorpack's :class:`HorovodTrainer`.
#           It uses the aggregated statistics of the whole batch (across all MPI ranks) to normalize.
#           Note that on single machine this is significantly slower than the "nccl" implementation.
#
#           If not None, per-GPU E[x] and E[x^2] among all GPUs are averaged to compute
#           global mean & variance. Therefore each GPU needs to have the same batch size.
#
#           The BatchNorm layer on each GPU needs to use the same name (`BatchNorm('name', input)`), so that
#           statistics can be reduced. If names do not match, this layer will hang.
#
#           This option only has effect in standard training mode.
#
#           This option is also known as "Cross-GPU BatchNorm" as mentioned in:
#           `MegDet: A Large Mini-Batch Object Detector <https://arxiv.org/abs/1711.07240>`_.
#           Corresponding TF issue: https://github.com/tensorflow/tensorflow/issues/18222.
#
#     Variable Names:
#
#     * ``beta``: the bias term. Will be zero-inited by default.
#     * ``gamma``: the scale term. Will be one-inited by default.
#     * ``mean/EMA``: the moving average of mean.
#     * ``variance/EMA``: the moving average of variance.
#
#     Note:
#         Combinations of ``training`` and ``ctx.is_training``:
#
#         * ``training == ctx.is_training``: standard BN, EMA are maintained during training
#           and used during inference. This is the default.
#         * ``training and not ctx.is_training``: still use batch statistics in inference.
#         * ``not training and ctx.is_training``: use EMA to normalize in
#           training. This is useful when you load a pre-trained BN and
#           don't want to fine tune the EMA. EMA will not be updated in
#           this case.
#     """
#     # parse shapes
#     shape = inputs.get_shape().as_list()
#     ndims = len(shape)
#     assert ndims in [2, 4], ndims
#
#     if axis is None:
#         if ndims == 2:
#             axis = 1
#         else:
#             axis = 1 if data_format == 'channels_first' else 3
#     assert axis in [1, 3], axis
#     num_chan = shape[axis]
#
#     red_axis = [0] if ndims == 2 else ([0, 2, 3] if axis == 1 else [0, 1, 2])
#
#     batch_mean = tf.reduce_mean(inputs, axis=red_axis)
#     batch_mean_square = tf.reduce_mean(tf.square(inputs), axis=red_axis)
#
#     if sync_statistics:
#         # Require https://github.com/uber/horovod/pull/331
#         import horovod_estimator.tensorflow as hvd
#         if hvd.size() != 1:
#             import horovod_estimator
#             hvd_version = tuple(map(int, horovod_estimator.__version__.split('.')))
#             assert hvd_version >= (0, 13, 6), "sync_statistics=horovod_estimator needs horovod_estimator>=0.13.6 !"
#
#             lp_debug('begin to allreduce mean and square')
#             batch_mean = hvd.allreduce(batch_mean, average=True)
#             batch_mean_square = hvd.allreduce(batch_mean_square, average=True)
#             lp_debug('end to allreduce mean and square')
#
#     batch_var = batch_mean_square - tf.square(batch_mean)
#     beta, gamma, moving_mean, moving_var = get_bn_variables(num_chan, scale, center, beta_initializer,
#                                                             gamma_initializer)
#     xn = tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, gamma, epsilon)
#     ret = update_bn_ema(xn, batch_mean, batch_var, moving_mean, moving_var, momentum, internal_update)
#     return ret
