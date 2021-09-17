from collections import defaultdict

import tensorflow as tf
from tensorflow.python.ops import init_ops

_BATCH_NORM_DECAY = 0.997
_BATCH_NORM_EPSILON = 1e-5


def float32_variable_storage_getter(getter, name, shape=None, dtype=None, initializer=None, regularizer=None, trainable=True, *args, **kwargs):
    storage_dtype = tf.float32 if trainable else dtype
    variable = getter(name, shape, dtype=storage_dtype,
                      initializer=initializer, regularizer=regularizer,
                      trainable=trainable,
                      *args, **kwargs)
    if trainable and dtype != tf.float32:
        variable = tf.cast(variable, dtype)
    return variable


class NNFunction(object):
    def __init__(self, is_training=True, dtype=tf.float32, data_format='NCHW', batch_norm_decay=_BATCH_NORM_DECAY, batch_norm_epsilon=_BATCH_NORM_EPSILON):
        self._is_training = is_training
        self._dtype = dtype
        self._batch_norm_decay = batch_norm_decay
        self._batch_norm_epsilon = batch_norm_epsilon
        self._data_format = data_format
        self._net_points = {}
        self._layer_counts = defaultdict(lambda: 0)

    @property
    def dtype(self):
        return self._dtype

    @property
    def data_foramt(self):
        return self._data_format

    def _ajust_with_data_foramt(self, x):
        assert len(x) == 4
        if self._data_format == 'NHWC':
            return [x[0], x[2], x[3], x[1]]
        else:
            return x

    def _get_in_channels(self, x):
        return x.get_shape().as_list()[1 if self._data_format == 'NCHW' else -1]

    def _count_layer(self, layer_type):
        idx = self._layer_counts[layer_type]
        name = layer_type + str(idx)
        self._layer_counts[layer_type] += 1
        return name

    def batch_norm(self, inputs, name='bn'):
        return tf.layers.batch_normalization(
            inputs=inputs, axis=1 if self._data_format == 'NCHW' else 3,
            momentum=self._batch_norm_decay, epsilon=self._batch_norm_epsilon, center=True,
            scale=True, training=self._is_training, fused=True, name=name)

    def _get_variable(self, name, shape, dtype=None, initializer=None, seed=None):
        if dtype is None:
            dtype = self._dtype
        if initializer is None:
            initializer = init_ops.glorot_uniform_initializer(seed=seed)
        return tf.get_variable(name, shape, dtype, initializer)

    def conv2d(self, x, output_channels, filter_size, filter_strides=(1, 1), padding='SAME', use_biases=True, name=None):
        kernel_shape = [filter_size[0], filter_size[1], self._get_in_channels(x), output_channels]
        strides = self._ajust_with_data_foramt([1, 1, filter_strides[0], filter_strides[1]])

        if name is None:
            name = self._count_layer('conv2d')

        with tf.variable_scope(name):
            kernel = self._get_variable('weights', kernel_shape, x.dtype)

            if padding == 'SAME_RESNET':  # ResNet models require custom padding
                kh, kw = filter_size
                rate = 1
                kernel_size_effective = kh + (kw - 1) * (rate - 1)
                pad_total = kernel_size_effective - 1
                pad_beg = pad_total // 2
                pad_end = pad_total - pad_beg
                padding = self._ajust_with_data_foramt([[0, 0], [0, 0], [pad_beg, pad_end], [pad_beg, pad_end]])
                x = tf.pad(x, padding)
                padding = 'VALID'

            x = tf.nn.conv2d(x, kernel, strides, padding=padding, data_format=self._data_format)

            if use_biases:
                biases = self._get_variable('biases', [self._get_in_channels(x)], x.dtype, initializer=tf.constant_initializer(0.0))
                if len(x.get_shape()) == 4:
                    return tf.nn.bias_add(x, biases, data_format=self._data_format)
                else:
                    return x + biases
            else:
                return x

    def conv2d_bn(self, x, num_filters, filter_size, filter_strides=(1, 1), padding='SAME', name=None):
        if name is None:
            name = self._count_layer('conv2d_bn')
        with tf.variable_scope(name):
            x = self.conv2d(x, num_filters, filter_size, filter_strides=filter_strides, padding=padding, use_biases=False)
            x = self.batch_norm(x)
        return x

    def max_pool2d(self, x, window_size, window_strides, padding='SAME', name='max_pool'):
        kernel_size = self._ajust_with_data_foramt([1, 1, window_size[0], window_size[1]])
        kernel_strides = self._ajust_with_data_foramt([1, 1, window_strides[0], window_strides[1]])
        return tf.nn.max_pool(x, kernel_size, kernel_strides, padding=padding, data_format=self._data_format, name=name)

    def avg_pool2d(self, x, window_size, window_strides, padding='SAME', name='avg_pool'):
        kernel_size = self._ajust_with_data_foramt([1, 1, window_size[0], window_size[1]])
        kernel_strides = self._ajust_with_data_foramt([1, 1, window_strides[0], window_strides[1]])
        return tf.nn.avg_pool(x, kernel_size, kernel_strides, padding=padding, data_format=self._data_format, name=name)

    def relu(self, x):
        return tf.nn.relu(x)

    def spatial_avg(self, x):
        dims = [2, 3] if self._data_format == 'NCHW' else [1, 2]
        return tf.reduce_mean(x, dims, keepdims=True, name='spatial_avg')

    def squeeze_hw(self, x, name=None):
        axis = [2, 3] if self._data_format == 'NCHW' else [1, 2]
        return tf.squeeze(x, axis=axis, name=name)


def resnet_bottleneck_v1_A(F: NNFunction, x, depth, bottleneck_depth, stride, downsample):
    s = stride

    if downsample:
        shortcut = F.conv2d_bn(x, depth, (1, 1), (s, s), padding='SAME', name='downsample')
    else:
        shortcut = x

    x = F.conv2d_bn(x, bottleneck_depth, (1, 1), (s, s), padding='SAME', name='conv2d_bn1')
    x = F.conv2d_bn(x, bottleneck_depth, (3, 3), (1, 1), padding='SAME', name='conv2d_bn2')
    x = F.conv2d_bn(x, depth, (1, 1), padding='SAME', name='conv2d_bn3')

    x = F.relu(x + shortcut)
    return x


def resnet_bottleneck_v1_B(F: NNFunction, x, depth, bottleneck_depth, stride, downsample):
    s = stride

    if downsample:
        shortcut = F.conv2d_bn(x, depth, (1, 1), (s, s), padding='SAME', name='downsample')
    else:
        shortcut = x

    x = F.conv2d_bn(x, bottleneck_depth, (1, 1), padding='SAME', name='conv2d_bn1')
    x = F.conv2d_bn(x, bottleneck_depth, (3, 3), (s, s), padding='SAME_RESNET', name='conv2d_bn2')
    x = F.conv2d_bn(x, depth, (1, 1), padding='SAME', name='conv2d_bn3')

    x = F.relu(x + shortcut)
    return x


def resnet_bottleneck_v1_D(F: NNFunction, x, depth, bottleneck_depth, stride, downsample):
    s = stride

    if downsample:
        shortcut = F.avg_pool2d(x, (2, 2), (s, s), padding='SAME')
        shortcut = F.conv2d_bn(shortcut, depth, (1, 1), padding='SAME', name='downsample')
    else:
        shortcut = x

    x = F.conv2d_bn(x, bottleneck_depth, (1, 1), padding='SAME', name='conv2d_bn1')
    x = F.conv2d_bn(x, bottleneck_depth, (3, 3), (s, s), padding='SAME_RESNET', name='conv2d_bn2')
    x = F.conv2d_bn(x, depth, (1, 1), padding='SAME', name='conv2d_bn3')

    x = F.relu(x + shortcut)
    return x


resnet_bottleneck_map = {
    'A': resnet_bottleneck_v1_A,
    'B': resnet_bottleneck_v1_B,
    'C': resnet_bottleneck_v1_B,
    'D': resnet_bottleneck_v1_D
}


def resnext_split_branch(net, input_layer, stride):
    x = input_layer
    with tf.name_scope('resnext_split_branch'):
        x = net.conv(x, net.bottleneck_width, (1, 1), (stride, stride), activation='RELU', use_batch_norm=True)
        x = net.conv(x, net.bottleneck_width, (3, 3), (1, 1), activation='RELU', use_batch_norm=True)
    return x


def resnext_shortcut(net, input_layer, stride, input_size, output_size):
    x = input_layer
    useConv = net.shortcut_type == 'C' or (net.shortcut_type == 'B' and input_size != output_size)
    with tf.name_scope('resnext_shortcut'):
        if useConv:
            x = net.conv(x, output_size, (1, 1), (stride, stride), use_batch_norm=True)
        elif output_size == input_size:
            if stride == 1:
                x = input_layer
            else:
                x = net.pool(x, 'MAX', (1, 1), (stride, stride))
        else:
            x = input_layer
    return x


def resnext_bottleneck_v1(net: NNFunction, x, depth, depth_bottleneck, stride):
    num_inputs = x.get_shape().as_list()[1]
    x = x
    with tf.name_scope('resnext_bottleneck_v1'):
        shortcut = resnext_shortcut(net, x, stride, num_inputs, depth)
        branches_list = []
        for i in range(net.cardinality):
            branch = resnext_split_branch(net, x, stride)
            branches_list.append(branch)
        concatenated_branches = tf.concat(values=branches_list, axis=1, name='concat')
        x = net.conv(concatenated_branches, depth, (1, 1), (1, 1), activation=None)
        x = net.activate(x + shortcut, 'RELU')
    return x


def resnet_v1(kind: str, F: NNFunction, layer_counts, num_classes, x):
    assert F.dtype == x.dtype, 'model dtype {} != input dtype {}'.format(F.dtype, x.dtype)

    net_points = {}
    with tf.variable_scope('resnet', custom_getter=float32_variable_storage_getter):
        if kind == 'A':
            x = F.conv2d(x, 64, [7, 7], filter_strides=[2, 2], padding='SAME_RESNET', name='conv1')
        else:
            x = F.relu(F.conv2d_bn(x, 32, [3, 3], filter_strides=[2, 2], padding='SAME_RESNET', name='conv1_1'))
            x = F.relu(F.conv2d_bn(x, 32, [3, 3], filter_strides=[1, 1], padding='SAME', name='conv1_2'))
            x = F.relu(F.conv2d_bn(x, 64, [3, 3], filter_strides=[1, 1], padding='SAME', name='conv1_3'))

        x = F.relu(F.batch_norm(x, name='bn1'))
        x = F.max_pool2d(x, [3, 3], [2, 2], padding='SAME', name='max_pool1')

        first_block_filters = 64
        resnet_bottleneck = resnet_bottleneck_map[kind]
        for i, count in enumerate(layer_counts):
            bottleneck_channels = first_block_filters * (2 ** i)
            output_channels = bottleneck_channels * 4
            with tf.variable_scope('layer{}'.format(i + 1)):
                for j in range(count):
                    with tf.variable_scope('block{}'.format(j + 1)):
                        x = resnet_bottleneck(F, x, output_channels, bottleneck_channels, stride=2 if j == 0 and i != 0 else 1,
                                              downsample=True if j == 0 else False)

        feature_map = x
        net_points['feature_map'] = feature_map
        x = F.spatial_avg(x)

        with tf.variable_scope("fc_weight", reuse=tf.AUTO_REUSE):
            x = F.conv2d(x, num_classes, (1, 1), name='logits')
            heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), name='logits')

        x = F.squeeze_hw(x)
        net_points['logits'] = x
        net_points['HeatMapFeatures'] = heat_map_features

        return x, net_points


choices = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}

if __name__ == '__main__':

    def sv():
        for i in tf.global_variables():
            print(i)


    tf.reset_default_graph()
    model = resnet_v1_50_D(1001, is_training=True, dtype=tf.float16, data_format='NHWC')

    img = tf.ones([1, 224, 224, 3], dtype=tf.float16)
    x = img
    logits, net_points = model(x)


    def exclude_batch_norm_and_bias(name):
        for i in ['gamma', 'beta', 'batch_normalization', 'BatchNorm', 'biases']:
            if i in name:
                return False
        return True


    trainable_variables_without_bn = [v for v in tf.trainable_variables() if exclude_batch_norm_and_bias(v.name)]
    for i in trainable_variables_without_bn:
        print(i)
