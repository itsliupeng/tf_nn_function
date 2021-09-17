from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import init_ops

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5

'''
default: padding='SAME', data_format='HCHW', dtype=tf.float16, is_training=True
'''


def float32_variable_storage_getter(getter,
                                    name,
                                    shape=None,
                                    dtype=None,
                                    initializer=None,
                                    regularizer=None,
                                    trainable=True,
                                    *args,
                                    **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name,
                      shape,
                      dtype=storage_dtype,
                      initializer=initializer,
                      regularizer=regularizer,
                      trainable=trainable,
                      *args,
                      **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


class F_scope:
    def __init__(self, name):
        self._name = name
        if self._name == '' or self._name is None:
            self._tf_scope = None
        else:
            self._tf_scope = tf.variable_scope(name)

    def __enter__(self):
        if self._tf_scope is not None:
            self._tf_scope.__enter__()

    def __exit__(self, type_arg, value_arg, traceback_arg):
        if self._tf_scope is not None:
            self._tf_scope.__exit__(type_arg, value_arg, traceback_arg)


class NNFunction(object):
    def __init__(self,
                 is_training=True,
                 dtype=tf.float32,
                 data_format='NCHW',
                 padding='SAME',
                 batch_norm_decay=_BATCH_NORM_DECAY,
                 batch_norm_epsilon=_BATCH_NORM_EPSILON):
        self._is_training = is_training
        self._dtype = dtype
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._data_format = data_format
        self._padding = padding
        self._net_points = {}
        self._layer_counts = defaultdict(lambda: 0)

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_foramt(self):
        return self._data_format

    @property
    def is_training(self):
        return self._is_training

    @property
    def channel_axis(self):
        return 1 if self._data_format == 'NCHW' else -1

    def _adjust_with_data_format(self, x):
        # default is NCHW
        assert len(x) == 4
        if self._data_format == 'NHWC':
            return [x[0], x[2], x[3], x[1]]
        else:
            return x

    def set_training(self, is_training):
        self._is_training = is_training
        return self

    def get_channels(self, x):
        return x.shape.as_list()[1 if self._data_format == 'NCHW' else -1]

    def count_layer(self, layer_type, name_suffix=None):
        name = '{}_{}'.format(layer_type, self._layer_counts[layer_type])
        if name_suffix is not None:
            name += name_suffix
        self._layer_counts[layer_type] += 1
        return name

    def batch_norm(self, inputs, name='bn'):
        return tf.layers.batch_normalization(
            inputs=inputs,
            axis=1 if self._data_format == 'NCHW' else 3,
            momentum=self._batch_norm_decay,
            epsilon=self._batch_norm_epsilon,
            center=True,
            scale=True,
            training=self._is_training,
            fused=True,
            name=name)

    def _get_variable(self, name, shape, dtype=None, initializer=None, seed=None):
        if dtype is None:
            dtype = self._dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        return tf.get_variable(name, shape, dtype, initializer)

    def conv2d(self,
               x,
               out_channels,
               filter_size,
               filter_strides=(1, 1),
               padding=None,
               use_biases=True,
               kernel_initializer=None,
               bias_initializer=None,
               num_groups=1,
               name=None):
        in_channels = self.get_channels(x)

        assert in_channels % num_groups == 0 and out_channels % num_groups == 0
        kernel_shape = [filter_size[0], filter_size[1], in_channels, out_channels // num_groups]

        strides = self._adjust_with_data_format([1, 1, filter_strides[0], filter_strides[1]])
        padding = padding if padding is not None else self._padding

        with F_scope(name):
            kernel = self._get_variable('kernel', kernel_shape, x.dtype, initializer=kernel_initializer)

            if padding == 'SAME_RESNET':  # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1  # rate is a stub for dilated conv
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                padding = self._adjust_with_data_format([[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
                x = tf.pad(x, padding)
                padding = 'VALID'

            if num_groups == 1:
                x = tf.nn.conv2d(x, kernel, strides, padding=padding, data_format=self._data_format)
            else:
                x_groups = tf.split(x, num_groups, axis=self.channel_axis)
                kernel_groups = tf.split(kernel, num_groups, axis=-2)
                out_groups = []
                for x_i, kernel_i in zip(x_groups, kernel_groups):
                    out_groups.append(tf.nn.conv2d(x_i, kernel_i, strides, padding=padding, data_format=self._data_format))
                x = tf.concat(out_groups, axis=self.channel_axis)

            if use_biases:
                if bias_initializer is None:
                    bias_initializer = tf.zeros_initializer()
                biases = self._get_variable('bias', [self.get_channels(x)], x.dtype, initializer=bias_initializer)
                if len(x.get_shape()) == 4:
                    return tf.nn.bias_add(x, biases, data_format=self._data_format)
                else:
                    return x + biases
            else:
                return x

    def conv2d_bn(self,
                  x,
                  out_channels,
                  filter_size,
                  filter_strides=(1, 1),
                  padding=None,
                  kernel_initializer=None,
                  bias_initializer=None,
                  num_groups=1,
                  name=None):
        with F_scope(name):
            x = self.conv2d(x,
                            out_channels,
                            filter_size,
                            filter_strides=filter_strides,
                            padding=padding,
                            use_biases=False,
                            kernel_initializer=kernel_initializer,
                            bias_initializer=bias_initializer,
                            num_groups=num_groups,
                            name='')
            x = self.batch_norm(x)
        return x

    def conv2d_bn_relu(self,
                       x,
                       out_channels,
                       filter_size,
                       filter_strides=(1, 1),
                       padding=None,
                       kernel_initializer=None,
                       bias_initializer=None,
                       name=None,
                       num_groups=1):
        with F_scope(name):
            x = self.conv2d_bn(x,
                               out_channels,
                               filter_size,
                               filter_strides,
                               padding=padding,
                               kernel_initializer=kernel_initializer,
                               bias_initializer=bias_initializer,
                               num_groups=num_groups,
                               name='')
            x = tf.nn.relu(x)
        return x

    def conv2d_bn_relu6(self, x, out_channels, filter_size, filter_strides=(1, 1), padding=None,
                        kernel_initializer=None, name=None, num_groups=1):
        with F_scope(name):
            x = self.conv2d_bn(x,
                               out_channels,
                               filter_size,
                               filter_strides,
                               padding=padding,
                               kernel_initializer=kernel_initializer,
                               num_groups=num_groups)
            x = tf.nn.relu6(x)
        return x

    def max_pool2d(self, x, window_size, window_strides=(1, 1), padding=None, name='MaxPool'):
        kernel_size = self._adjust_with_data_format([1, 1, window_size[0], window_size[1]])
        kernel_strides = self._adjust_with_data_format([1, 1, window_strides[0], window_strides[1]])
        padding = padding if padding is not None else self._padding
        return tf.nn.max_pool(x, kernel_size, kernel_strides, padding=padding, data_format=self._data_format, name=name)

    def avg_pool2d(self, x, window_size, window_strides=(1, 1), padding=None, name='AvgPool'):
        kernel_size = self._adjust_with_data_format([1, 1, window_size[0], window_size[1]])
        kernel_strides = self._adjust_with_data_format([1, 1, window_strides[0], window_strides[1]])
        padding = padding if padding is not None else self._padding
        return tf.nn.avg_pool(x, kernel_size, kernel_strides, padding=padding, data_format=self._data_format, name=name)

    def relu(self, x, name=None):
        return tf.nn.relu(x, name)

    def relu6(self, x, name=None):
        return tf.nn.relu6(x, name)

    def dropout(self, x, rate=0.5, is_training=None, noise_shape=None, seed=None, name=None):
        is_training = is_training if is_training is not None else self._is_training
        if is_training:
            return tf.nn.dropout(x, rate=rate, noise_shape=noise_shape, seed=seed, name=name)
        else:
            return x

    def spatial_avg(self, x):
        dims = [2, 3] if self._data_format == 'NCHW' else [1, 2]
        return tf.reduce_mean(x, dims, keepdims=True, name='SpatialAvg')

    def squeeze_hw(self, x, name=None):
        axis = [2, 3] if self._data_format == 'NCHW' else [1, 2]
        return tf.squeeze(x, axis=axis, name=name)

    def depthwise_conv2d(self, x, channel_multiplier, filter_size, filter_strides=(1, 1), padding=None, name=None):
        in_channels = self.get_channels(x)
        kernel_shape = [filter_size[0], filter_size[1], in_channels, channel_multiplier]
        strides = self._adjust_with_data_format([1, 1, filter_strides[0], filter_strides[1]])

        with F_scope(name):
            kernel = self._get_variable('kernel', kernel_shape, x.dtype)
            padding = padding if padding is not None else self._padding
            x = tf.nn.depthwise_conv2d(x, kernel, strides, padding=padding, data_format=self._data_format)

        return x

    def depthwise_conv2d_bn(self, x, channel_multiplier, filter_size, filter_strides=(1, 1), padding=None, name=None):
        with F_scope(name):
            x = self.depthwise_conv2d(x, channel_multiplier, filter_size, filter_strides, padding)
            x = self.batch_norm(x)

        return x

    def depthwise_conv2d_bn_relu(self, x, channel_multiplier, filter_size, filter_strides=(1, 1), padding=None,
                                 name=None):
        with F_scope(name):
            x = self.depthwise_conv2d_bn(x, channel_multiplier, filter_size, filter_strides, padding)
            x = self.relu(x)

        return x

    def depthwise_conv2d_bn_relu6(self, x, channel_multiplier, filter_size, filter_strides=(1, 1), padding=None,
                                  name=None):
        with F_scope(name):
            x = self.depthwise_conv2d_bn(x, channel_multiplier, filter_size, filter_strides, padding)
            x = self.relu6(x)

        return x

    def separable_conv2d(self, x, out_channels, filter_size, filter_strides=(1, 1), padding=None, channel_multiplier=1,
                         name=None):
        in_channels = self.get_channels(x)
        depthwise_kernel_shape = [filter_size[0], filter_size[1], in_channels, channel_multiplier]
        strides = self._adjust_with_data_format([1, 1, filter_strides[0], filter_strides[1]])
        padding = padding if padding is not None else self._padding
        pointwise_kernel_shape = [1, 1, channel_multiplier * in_channels, out_channels]

        with F_scope(name):
            depthwise_kernel = self._get_variable('depthwise_kernel', depthwise_kernel_shape, x.dtype)
            pointwise_kernel = self._get_variable('pointwise_kernel', pointwise_kernel_shape, x.dtype)
            x = tf.nn.separable_conv2d(x, depthwise_kernel, pointwise_kernel, strides, padding=padding,
                                       data_format=self._data_format)

        return x

    def separable_conv2d_bn(self, x, out_channels, filter_size, filter_strides=(1, 1), padding=None,
                            channel_multiplier=1, name=None):
        with F_scope(name):
            x = self.separable_conv2d(x, out_channels, filter_size, filter_strides, padding=padding,
                                      channel_multiplier=channel_multiplier)
            x = self.batch_norm(x)

        return x

    def separable_conv2d_bn_relu6(self, x, out_channels, filter_size, filter_strides=(1, 1), padding=None,
                                  channel_multiplier=1, name=None):
        with F_scope(name):
            x = self.separable_conv2d_bn(x, out_channels, filter_size, filter_strides, padding=padding,
                                         channel_multiplier=channel_multiplier)
            x = self.relu6(x)

        return x

    def channel_shuffle(self, x, num_group):
        with tf.variable_scope('ChannelShuffle'):
            if self.data_foramt == 'NHWC':
                n, h, w, c = x.get_shape().as_list()
                assert c % num_group == 0
                x = tf.reshape(x, shape=[-1, h, w, num_group, c // num_group])
                x = tf.transpose(x, [0, 1, 2, 4, 3])
                x = tf.reshape(x, shape=[-1, h, w, c])
            else:
                n, c, h, w = x.get_shape().as_list()
                assert c % num_group == 0
                x = tf.reshape(x, shape=[-1, num_group, c // num_group, h, w])
                x = tf.transpose(x, [0, 2, 1, 3, 4])
                x = tf.reshape(x, shape=[-1, c, h, w])

        return x

    def split(self, x, num_splits):
        assert self.get_channels(x) % num_splits == 0
        return tf.split(x, num_splits, axis=self.channel_axis)

    def se(self, x, ratio=1 / 16, num_reduced_filters=None, name=None):
        out_channels = self.get_channels(x)
        if num_reduced_filters is None:
            num_reduced_filters = int(ratio * out_channels)

        with F_scope(name):
            squeeze = self.spatial_avg(x)
            squeeze = self.conv2d(squeeze, num_reduced_filters, (1, 1), use_biases=True, name='FC_0')
            squeeze = tf.nn.relu(squeeze)
            excitation = self.conv2d(squeeze, out_channels, (1, 1), use_biases=True, name='FC_1')
            excitation = tf.nn.sigmoid(excitation)

            return x * excitation

    def fc(self, x, out_channels, use_biases=True, kernel_initializer=None, bias_initializer=None, name=None):
        with F_scope(name):
            assert len(x.shape.dims) == 2
            if self._data_format == 'NCHW':
                x = tf.expand_dims(tf.expand_dims(x, -1), -1)  # not use tf.reshape to keep prevent placeholder None dimension bug when export to pb
            else:
                x = tf.expand_dims(tf.expand_dims(x, 1), 1)

            x = self.conv2d(x, out_channels, (1, 1), use_biases=use_biases, kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
            return tf.reshape(x, [-1, out_channels])

    def drop_block(self, x, keep_prob=None, block_size=7, is_training=None):
        is_training = is_training if is_training is not None else self._is_training

        if not is_training:
            return x

        tf.logging.info('Applying DropBlock: keep_prob {},  dropblock_size {}, x.shape {}'.format(keep_prob, block_size, x.shape))

        if self._data_format == 'NCHW':
            _, _, height, width = x.shape.as_list()
        else:
            _, height, width, _ = x.shape.as_list()

        total_size = width * height
        dropblock_size = min(block_size, min(width, height))

        # Seed_drop_rate is the gamma parameter of DropBlcok.
        seed_drop_rate = (1.0 - keep_prob) * total_size / block_size ** 2 / ((width - block_size + 1) * (height - block_size + 1))

        # Forces the block to be inside the feature map.
        w_i, h_i = tf.meshgrid(tf.range(width), tf.range(height))
        valid_block = tf.logical_and(
            tf.logical_and(w_i >= int(dropblock_size // 2), w_i < width - (dropblock_size - 1) // 2),
            tf.logical_and(h_i >= int(dropblock_size // 2), h_i < width - (dropblock_size - 1) // 2)
        )

        if self._data_format == 'NCHW':
            valid_block = tf.reshape(valid_block, [1, 1, height, width])
        else:
            valid_block = tf.reshape(valid_block, [1, height, width, 1])

        valid_block = tf.cast(valid_block, dtype=tf.float32)
        seed_keep_rate = tf.cast(1 - seed_drop_rate, dtype=tf.float32)

        rand_noise = tf.random_uniform(x.shape, dtype=tf.float32)
        block_pattern = (1 - valid_block + seed_keep_rate + rand_noise) >= 1
        block_pattern = tf.cast(block_pattern, dtype=tf.float32)

        block_pattern = -self.max_pool2d(-block_pattern, (block_size, block_size), padding='SAME')

        percent_ones = tf.cast(tf.reduce_sum(block_pattern), tf.float32) / tf.cast(tf.size(block_pattern), tf.float32)

        x = x * tf.cast(block_pattern, x.dtype) / tf.cast(percent_ones, x.dtype)
        return x


if __name__ == '__main__':
    import tensorflow.contrib.eager as tfe

    tfe.enable_eager_execution()

    F = NNFunction(data_format='NHWC')
    # x = tf.ones([1, 7, 7, 128])
    # y1 = F.conv2d(x, 256, (3, 3), num_groups=1, name='Conv2dG1')
    # y32 = F.conv2d(x, 256, (3, 3), num_groups=32, name='Conv2dG32')

    xx = tf.ones([2, 10, 10, 3])
    yy = F.drop_block(xx, keep_prob=0.9)

    print('Done')
