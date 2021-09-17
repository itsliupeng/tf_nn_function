from functools import partial

import tensorflow as tf

from models.nn_functions import NNFunction, float32_variable_storage_getter


def bottleneck(F: NNFunction, x, out_channels, s):
    with tf.variable_scope(F.count_layer('Bottleneck')):
        x = F.depthwise_conv2d_bn_relu(x, 1, (3, 3), (s, s), name='DwConv2dS{}_Bn_Relu'.format(s))
        x = F.conv2d_bn_relu(x, out_channels, (1, 1), name='Conv2d_Bn_Relu')

    return x


def _make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


bottleneck_cfg = [(32, 64, 1),
                  (64, 128, 2),
                  (128, 128, 1),
                  (128, 256, 2),
                  (256, 256, 1),
                  (256, 512, 2)] + \
                 [(512, 512, 1)] * 5 + \
                 [(512, 1024, 2),
                  (1024, 1024, 1)]


def mobilenet_v1(num_classes, is_training, dtype=tf.float16, data_format='NCHW', multiplier=1.0):
    F = NNFunction(is_training, dtype, data_format)

    def mobilenet_v2_fn(x):
        with tf.variable_scope('MobileNetV1', custom_getter=float32_variable_storage_getter):
            end_points = {}

            x = F.conv2d_bn_relu6(x, _make_divisible(int(32 * multiplier)), (3, 3), (2, 2), name='initial_conv')

            for i, (_, out_channels, stride) in enumerate(bottleneck_cfg):
                out_channels = _make_divisible(int(out_channels * multiplier))
                x = bottleneck(F, x, out_channels, stride)

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


mobilenet_v1_1_0 = partial(mobilenet_v1, multiplier=1.0)
mobilenet_v1_0_5 = partial(mobilenet_v1, multiplier=0.5)

if __name__ == '__main__':

    tf.reset_default_graph()

    model = mobilenet_v1_0_5(1001, is_training=True, dtype=tf.float16, data_format='NHWC')
    img = tf.placeholder(tf.float16, [1, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [1, 1])
    _logits, _end_points = model(img)

    from models import show_model
    show_model()
