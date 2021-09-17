from functools import partial

import tensorflow as tf

from models.nn_functions import NNFunction, float32_variable_storage_getter


def shufflenet_v2_block_A(F: NNFunction, x, out_channels, stride, use_identity=False, use_se=False):
    with tf.variable_scope(F.count_layer('Block')):
        half_channels = out_channels // 2
        if stride == 1:
            out_left, out_right = F.split(x, 2)
            with tf.variable_scope('Right'):
                identity = out_right
                out_right = F.conv2d_bn_relu(out_right, half_channels, (1, 1), name='Conv2d_Bn_Relu')
                out_right = F.depthwise_conv2d_bn(out_right, 1, (3, 3), name='DwConv2d_Bn_Relu')
                out_right = F.conv2d_bn(out_right, half_channels, (1, 1), name='Conv2d_Bn')
                if use_se:
                    out_right = F.se(out_right, name='SE')
                if use_identity:
                    out_right = out_right + identity

                out_right = F.relu(out_right)

        elif stride == 2:
            with tf.variable_scope('Left'):
                out_left = F.depthwise_conv2d_bn(x, 1, (3, 3), (stride, stride), name='DwConv2dS{}_Bn'.format(stride))
                out_left = F.conv2d_bn_relu(out_left, half_channels, (1, 1), name='Conv2d_Bn_Relu')
            with tf.variable_scope('Right'):
                out_right = F.conv2d_bn_relu(x, half_channels, (1, 1), name='Conv2d_Bn_Relu_0')
                out_right = F.depthwise_conv2d_bn(out_right, 1, (3, 3), (stride, stride), name='DwConv2dS{}_Bn'.format(stride))
                out_right = F.conv2d_bn_relu(out_right, half_channels, (1, 1), name='Conv2d_Bn_Relu_1')
        else:
            raise RuntimeError('stride {} should in [1, 2]'.format(stride))

        out = tf.concat([out_left, out_right], axis=F.channel_axis)
        out = F.channel_shuffle(out, num_group=2)
        return out


def shufflenet_v2_block_B(F: NNFunction, x, out_channels, stride, use_identity=False, use_se=False):
    with tf.variable_scope(F.count_layer('shuffle_v2_block')):
        half_channels = out_channels // 2
        if stride == 1:
            out_left, out_right = F.split(x, 2)
            with tf.variable_scope('right'):
                identity = out_right
                out_right = F.conv2d_bn_relu(out_right, half_channels, (1, 1))
                out_right = F.depthwise_conv2d_bn(out_right, 1, (3, 3))
                out_right = F.conv2d_bn(out_right, half_channels, (1, 1))
                if use_se:
                    out_right = F.se(out_right, name='se')
                if use_identity:
                    out_right = out_right + identity

                out_right = F.relu(out_right)

        elif stride == 2:
            with tf.variable_scope('left'):
                out_left = F.avg_pool2d(x, (3, 3), (2, 2))
                out_left = F.conv2d_bn_relu(out_left, half_channels, (1, 1))
            with tf.variable_scope('right'):
                out_right = F.conv2d_bn_relu(x, half_channels, (1, 1))
                out_right = F.depthwise_conv2d_bn(out_right, 1, (3, 3), (2, 2))
                out_right = F.conv2d_bn_relu(out_right, half_channels, (1, 1))
        else:
            raise RuntimeError('stride {} should in [1, 2]'.format(stride))

        out = tf.concat([out_left, out_right], axis=F.channel_axis)
        out = F.channel_shuffle(out, num_group=2)
        return out


def shufflenet_v2_block_C(F: NNFunction, x, out_channels, stride, use_identity=False, use_se=False):
    with tf.variable_scope(F.count_layer('shuffle_v2_block')):
        half_channels = out_channels // 2
        if stride == 1:
            out_left, out_right = F.split(x, 2)
            with tf.variable_scope('right'):
                identity = out_right
                out_right = F.conv2d_bn_relu(out_right, half_channels, (1, 1))
                out_right = F.depthwise_conv2d_bn(out_right, 1, (3, 3))
                out_right = F.conv2d_bn(out_right, half_channels, (1, 1))
                if use_se:
                    out_right = F.se(out_right, name='se')
                if use_identity:
                    out_right += identity

                out_right = F.relu(out_right)

        elif stride == 2:
            with tf.variable_scope('left'):
                out_left = F.avg_pool2d(x, (3, 3), (2, 2))
                if F.get_channels(out_left) != half_channels:
                    out_left = F.conv2d_bn(out_left, half_channels, (1, 1))
                    identity = out_left
                    out_left = F.relu(out_left)
                else:
                    identity = out_left
            with tf.variable_scope('right'):
                out_right = F.conv2d_bn_relu(x, half_channels, (1, 1))
                out_right = F.depthwise_conv2d_bn(out_right, 1, (3, 3), (2, 2))
                out_right = F.conv2d_bn(out_right, half_channels, (1, 1))
                if use_se:
                    out_right = F.se(out_right, name='se')
                if use_identity:
                    out_right += identity

                out_right = F.relu(out_right)

        else:
            raise RuntimeError('stride {} should in [1, 2]'.format(stride))

        out = tf.concat([out_left, out_right], axis=F.channel_axis)
        out = F.channel_shuffle(out, num_group=2)
        return out


def _select_channel_size(model_scale):
    # [(out_channel, repeat_times), (out_channel, repeat_times), ...]
    if model_scale == 0.5:
        return [24, (48, 4), (96, 8), (192, 4), 1024]
    elif model_scale == 1.0:
        return [24, (116, 4), (232, 8), (464, 4), 1024]
    elif model_scale == 1.5:
        return [24, (176, 4), (352, 8), (704, 4), 1024]
    elif model_scale == 2.0 or model_scale == 'se_2.0':
        return [24, (244, 4), (488, 8), (976, 4), 2048]
    elif model_scale == '50' or model_scale == 'se_50':
        return [64, (244, 3), (488, 4), (976, 6), (1952, 3), 2048]
    elif model_scale == 'se_164':
        return [64, (340, 10), (680, 10), (1360, 23), (2720, 10), 2048]
    else:
        raise ValueError('Unsupported model size.')


shufflenet_block = {
    'A': shufflenet_v2_block_A,
    'B': shufflenet_v2_block_B,
    'C': shufflenet_v2_block_C
}


def shufflenet_v2(num_classes, is_training, dtype=tf.float32, data_format='NCHW', model_scale=1.0, block_type='A', use_identity=False, use_se=False):
    F = NNFunction(is_training, dtype, data_format)

    def fn(x):
        end_points = {}

        with tf.variable_scope('ShuffleNetV2', custom_getter=float32_variable_storage_getter):
            channel_sizes = _select_channel_size(model_scale)

            if model_scale == 'se_164':
                with tf.variable_scope('InitConv'):
                    x = F.conv2d_bn_relu(x, 64, (3, 3), (2, 2))
                    x = F.conv2d_bn_relu(x, 64, (3, 3))
                    x = F.conv2d_bn_relu(x, 128, (3, 3))
            else:
                x = F.conv2d_bn_relu(x, channel_sizes[0], (3, 3), (2, 2), name='InitConv')

            x = F.max_pool2d(x, (3, 3), (2, 2), name='MaxPool')
            for idx, (out_channels, repeat) in enumerate(channel_sizes[1:-1]):
                with tf.variable_scope('Layer_{}'.format(idx)):
                    for i in range(repeat):
                        # use_identity = False
                        # use_se = False
                        # if model_scale in ['50', 'se_164', 'se_50']:
                        #     use_identity = True
                        # if model_scale in ['se_164', 'se_2.0', 'se_50']:
                        #     use_se = True

                        x = shufflenet_block[block_type](F, x, out_channels, stride=2 if i == 0 else 1,
                                                         use_identity=use_identity, use_se=use_se)

            x = F.conv2d_bn_relu(x, channel_sizes[-1], (1, 1), name='UpChannel')

            feature_map = x
            end_points['FeatureMap'] = feature_map
            x = F.spatial_avg(x)
            with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):
                x = F.conv2d(x, num_classes, (1, 1), use_biases=True, name='Logits')
                heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), use_biases=True, name='Logits')
                x = F.squeeze_hw(x)
                end_points['HeatMapFeatures'] = heat_map_features

        return x, end_points

    return fn


res_shufflenet_v2_2_0_C = partial(shufflenet_v2, model_scale=2.0, block_type='C', use_identity=True)
shufflenet_v2_2_0_C = partial(shufflenet_v2, model_scale=2.0, block_type='C')
shufflenet_v2_1_0_C = partial(shufflenet_v2, model_scale=1.0, block_type='C')
shufflenet_v2_2_0_B = partial(shufflenet_v2, model_scale=2.0, block_type='B')
shufflenet_v2_1_0_B = partial(shufflenet_v2, model_scale=1.0, block_type='B')
shufflenet_v2_2_0 = partial(shufflenet_v2, model_scale=2.0)
shufflenet_v2_1_5 = partial(shufflenet_v2, model_scale=1.5)
shufflenet_v2_1_0 = partial(shufflenet_v2, model_scale=1.0)
shufflenet_v2_0_5 = partial(shufflenet_v2, model_scale=0.5)

shufflenet_v2_50 = partial(shufflenet_v2, model_scale='50')
se_shufflenet_v2_50 = partial(shufflenet_v2, model_scale='se_50', use_identity=True, use_se=True)
se_shufflenet_v2_164 = partial(shufflenet_v2, model_scale='se_164', use_identity=True, use_se=True)
se_shufflenet_v2_2_0 = partial(shufflenet_v2, model_scale='se_2.0', use_identity=True, use_se=True)

if __name__ == '__main__':
    tf.reset_default_graph()

    model = shufflenet_v2_1_0(1001, is_training=True, data_format='NHWC')
    img = tf.placeholder(tf.float32, [1, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [1, 1])
    _logits, _end_points = model(img)
    from models import show_model

    show_model()
