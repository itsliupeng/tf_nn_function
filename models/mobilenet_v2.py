from functools import partial

import tensorflow as tf

from models.nn_functions import NNFunction, float32_variable_storage_getter


def linear_bottleneck(F: NNFunction, x, in_channels, out_channels, t, stride, keep_in_out_channels_same_conv=True):
    with tf.variable_scope(F.count_layer('Bottleneck')):
        shortcut = x
        s = stride
        if keep_in_out_channels_same_conv or F.get_channels(x) != in_channels * t:
            x = F.conv2d_bn_relu6(x, in_channels * t, (1, 1), name='Conv2d_Bn_Relu6')
        x = F.depthwise_conv2d_bn_relu6(x, 1, (3, 3), (s, s), name='DwConv2dS{}_Bn_Relu6'.format(s))
        x = F.conv2d_bn(x, out_channels, (1, 1), name='Conv2d_Bn')

        if stride == 1 and in_channels == out_channels:
            x = x + shortcut

        return x


def separate_linear_bottleneck(F: NNFunction, x, in_channels, out_channels, t, stride,
                               keep_in_out_channels_same_conv=True):
    with tf.variable_scope(F.count_layer('Bottleneck')):
        shortcut = x
        s = stride

        if keep_in_out_channels_same_conv or F.get_channels(x) != in_channels * t:
            x = F.conv2d_bn_relu6(x, in_channels * t, (1, 1), name='Conv2d_Bn_Relu')

        x = F.separable_conv2d_bn(x, out_channels, (3, 3), (s, s), name='Sep_Conv2dS{}_Bn'.format(s))

        if stride == 1 and in_channels == out_channels:
            x = x + shortcut

        return x


bottleneck_map = {
    'A': linear_bottleneck,
    'S': separate_linear_bottleneck
}


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def mobilenet_v2(num_classes, is_training, dtype=tf.float16, data_format='NCHW', multiplier=1.0, kind='A',
                 keep_in_out_channels_same_conv=True):
    F = NNFunction(is_training, dtype, data_format)

    def mobilenet_v2_fn(x):
        with tf.variable_scope('MobileNetV2', custom_getter=float32_variable_storage_getter):
            end_points = {}

            x = F.conv2d_bn_relu6(x, _make_divisible(int(32 * multiplier)), (3, 3), (2, 2), name='initial_conv')

            in_channels_group = [_make_divisible(int(x * multiplier)) for x in
                                 [32] + [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3]
            out_channels_group = [_make_divisible(int(x * multiplier)) for x in
                                  [16] + [24] * 2 + [32] * 3 + [64] * 4 + [96] * 3 + [160] * 3 + [320]]
            ts = [1] + [6] * 16
            strides = [1, 2] * 2 + [1, 1, 2] + [1] * 6 + [2] + [1] * 3

            for i, (in_c, out_c, t, s) in enumerate(zip(in_channels_group, out_channels_group, ts, strides)):
                x = bottleneck_map[kind](F, x, in_c, out_c, t, s, keep_in_out_channels_same_conv)
            last_channels = int(1280 * multiplier) if multiplier > 1.0 else 1280
            x = F.conv2d_bn_relu6(x, last_channels, (1, 1))

            feature_map = x
            x = F.spatial_avg(x)
            with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
                x = F.conv2d(x, num_classes, (1, 1), use_biases=True, name='Logits')
                heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), use_biases=True, name='Logits')

            x = F.squeeze_hw(x)
            end_points['Logits'] = x
            end_points['HeatMapFeatures'] = heat_map_features

        return x, end_points

    return mobilenet_v2_fn


mobilenet_v2_1_4_merge = partial(mobilenet_v2, multiplier=1.4, keep_in_out_channels_same_conv=False)
mobilenet_v2_1_0_merge = partial(mobilenet_v2, multiplier=1.0, keep_in_out_channels_same_conv=False)

mobilenet_v2_1_4 = partial(mobilenet_v2, multiplier=1.4)
mobilenet_v2_1_0 = partial(mobilenet_v2, multiplier=1.0)
mobilenet_v2_0_75 = partial(mobilenet_v2, multiplier=0.75)
mobilenet_v2_0_5 = partial(mobilenet_v2, multiplier=0.5)
mobilenet_v2_0_25 = partial(mobilenet_v2, multiplier=0.25)

mobilenet_v2_1_4_S = partial(mobilenet_v2, multiplier=1.4, kind='S')
mobilenet_v2_1_0_S = partial(mobilenet_v2, multiplier=1.0, kind='S')
mobilenet_v2_0_75_S = partial(mobilenet_v2, multiplier=0.75, kind='S')
mobilenet_v2_0_5_S = partial(mobilenet_v2, multiplier=0.5, kind='S')
mobilenet_v2_0_25_S = partial(mobilenet_v2, multiplier=0.25, kind='S')

if __name__ == '__main__':
    tf.reset_default_graph()

    model = mobilenet_v2_1_0(1001, is_training=True, dtype=tf.float16, data_format='NHWC')
    img = tf.placeholder(tf.float16, [1, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [1, 1])
    _logits, _end_points = model(img)

    from models import show_model

    show_model()
