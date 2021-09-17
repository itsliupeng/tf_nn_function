import tensorflow as tf

from models.nn_functions import NNFunction, float32_variable_storage_getter


def resnet_bottleneck_v1_A(F: NNFunction,
                           x,
                           depth,
                           bottleneck_depth,
                           stride,
                           downsample,
                           use_se=False,
                           num_groups=1,
                           keep_in_out_channels_same_conv=True,
                           drop_block_keep_rate=None):
    s = stride
    if downsample:
        with tf.variable_scope('Shortcut'):
            shortcut = F.conv2d_bn(x, depth, (1, 1), (s, s), padding='SAME', name='Conv2dS{}_BN'.format(s))
    else:
        shortcut = x

    if drop_block_keep_rate is not None:
        shortcut = F.drop_block(shortcut, keep_prob=drop_block_keep_rate)

    # 1
    if keep_in_out_channels_same_conv or F.get_channels(x) != bottleneck_depth:
        x = F.conv2d_bn_relu(x, bottleneck_depth, (1, 1), (s, s), padding='SAME', name='Conv2dS{}_BN_Relu_0'.format(s))
        if drop_block_keep_rate is not None:
            x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    # 2
    x = F.conv2d_bn_relu(x, bottleneck_depth, (3, 3), (1, 1), num_groups=num_groups, padding='SAME_RESNET', name='Conv2dS1_BN_Relu_1')
    if drop_block_keep_rate is not None:
        x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    # 3
    x = F.conv2d_bn(x, depth, (1, 1), padding='SAME', name='Conv2dS1_BN_2')
    if drop_block_keep_rate is not None:
        x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    if use_se:
        x = F.se(x, name='SE')

    x = F.relu(x + shortcut)
    return x


def resnet_bottleneck_v1_B(F: NNFunction, x, depth, bottleneck_depth, stride, downsample, use_se=False, num_groups=1, keep_in_out_channels_same_conv=True,
                           drop_block_keep_rate=None):
    s = stride
    if downsample:
        with tf.variable_scope('ShortCut'):
            shortcut = F.conv2d_bn(x, depth, (1, 1), (s, s), padding='SAME', name='Conv2dS{}_BN'.format(s))
    else:
        shortcut = x

    if drop_block_keep_rate is not None:
        shortcut = F.drop_block(shortcut, keep_prob=drop_block_keep_rate)

    if keep_in_out_channels_same_conv or F.get_channels(x) != bottleneck_depth:
        x = F.conv2d_bn_relu(x, bottleneck_depth, (1, 1), name='Conv2dS1_BN_Relu_0')
        if drop_block_keep_rate is not None:
            x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    x = F.conv2d_bn_relu(x, bottleneck_depth, (3, 3), (s, s), num_groups=num_groups, padding='SAME_RESNET', name='Conv2dS{}_BN_Relu_1'.format(s))
    if drop_block_keep_rate is not None:
        x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    x = F.conv2d_bn(x, depth, (1, 1), padding='SAME', name='Conv2dS1_BN_2')
    if drop_block_keep_rate is not None:
        x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    if use_se:
        x = F.se(x, name='SE')

    x = F.relu(x + shortcut)
    return x


def resnet_bottleneck_v1_D(F: NNFunction, x, depth, bottleneck_depth, stride, downsample, use_se=False, num_groups=1,
                           keep_in_out_channels_same_conv=True, drop_block_keep_rate=None):
    s = stride
    if downsample:
        with tf.variable_scope('Shortcut'):
            shortcut = F.avg_pool2d(x, (s, s), (s, s), padding='VALID', name='AvgPool')
            shortcut = F.conv2d_bn(shortcut, depth, (1, 1), name='Conv2dS1_BN')
    else:
        shortcut = x

    if drop_block_keep_rate is not None:
        shortcut = F.drop_block(shortcut, keep_prob=drop_block_keep_rate)

    # 1
    if keep_in_out_channels_same_conv or F.get_channels(x) != bottleneck_depth:
        x = F.conv2d_bn_relu(x, bottleneck_depth, (1, 1), name='Conv2dS1_BN_Relu_0')
        if drop_block_keep_rate is not None:
            x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    # 2
    x = F.conv2d_bn_relu(x, bottleneck_depth, (3, 3), (s, s), num_groups=num_groups, padding='SAME_RESNET', name='Conv2dS{}_BN_Relu_1'.format(s))
    if drop_block_keep_rate is not None:
        x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    # 3
    x = F.conv2d_bn(x, depth, (1, 1), name='Conv2dS1_BN_2')
    if drop_block_keep_rate is not None:
        x = F.drop_block(x, keep_prob=drop_block_keep_rate)

    if use_se:
        x = F.se(x, name='SE')

    x = F.relu(x + shortcut)
    return x


def linear_resnet_bottleneck_v1_D(F: NNFunction, x, depth, bottleneck_depth, stride, downsample, use_se=False,
                                  num_groups=1, keep_in_out_channels_same_conv=True):
    s = stride

    if downsample:
        shortcut = F.avg_pool2d(x, (s, s), (s, s), padding='VALID')
        shortcut = F.conv2d_bn(shortcut, depth, (1, 1), name='downsample')
    else:
        shortcut = x

    if F.get_channels(x) != bottleneck_depth:
        x = F.conv2d_bn(x, bottleneck_depth, (1, 1), name='1_conv2d_bn')
    x = F.conv2d_bn(x, bottleneck_depth, (3, 3), (s, s), padding='SAME_RESNET', num_groups=num_groups,
                    name='2_group{}_conv2d_bn'.format(num_groups))
    x = F.conv2d_bn(x, depth, (1, 1), name='3_conv2d_bn')

    if use_se:
        x = F.se(x, name='se')

    x = F.relu(x + shortcut)
    return x


bottleneck_map = {
    'A': resnet_bottleneck_v1_A,
    'B': resnet_bottleneck_v1_B,
    'C': resnet_bottleneck_v1_B,
    'D': resnet_bottleneck_v1_D,
    'L': linear_resnet_bottleneck_v1_D
}


def resnet_v1(kind: str,
              F: NNFunction,
              layer_counts,
              num_classes,
              x,
              use_se=False,
              num_groups=1,
              keep_in_out_channels_same_conv=True,
              return_layer=None,
              drop_block_keep_rate=None):
    assert F.dtype == x.dtype, 'model dtype {} != input dtype {}'.format(F.dtype, x.dtype)

    end_points = {}
    with tf.variable_scope('ResNet', custom_getter=float32_variable_storage_getter):
        with tf.variable_scope('InitConv'):
            if kind in ['A', 'B']:
                x = F.conv2d_bn_relu(x, 64, [7, 7], [2, 2], padding='SAME_RESNET', name='Conv2dS2_BN_Relu')
            else:
                x = F.conv2d_bn_relu(x, 32, [3, 3], [2, 2], padding='SAME_RESNET', name='Conv2dS2_BN_Relu_0')
                x = F.conv2d_bn_relu(x, 32, [3, 3], name='Conv2dS1_BN_Relu_1')
                x = F.conv2d_bn_relu(x, 64, [3, 3], name='Conv2dS1_BN_Relu_2')

        x = F.max_pool2d(x, [3, 3], [2, 2], name='MaxPool')

        first_block_filters = 64
        resnet_bottleneck = bottleneck_map[kind]
        for i, count in enumerate(layer_counts):
            bottleneck_channels = first_block_filters * (2 ** i)
            output_channels = bottleneck_channels * 4
            layer_name = 'Layer_{}'.format(i)
            with tf.variable_scope(layer_name):
                for j in range(count):
                    with tf.variable_scope('Block_{}'.format(j)):
                        x = resnet_bottleneck(F,
                                              x,
                                              output_channels,
                                              bottleneck_channels,
                                              stride=2 if j == 0 and i != 0 else 1,
                                              downsample=True if j == 0 else False,
                                              use_se=use_se,
                                              num_groups=num_groups,
                                              keep_in_out_channels_same_conv=keep_in_out_channels_same_conv,
                                              drop_block_keep_rate=drop_block_keep_rate if i >= 2 else None)
            # indent attention
            end_points[layer_name] = x
            if layer_name == return_layer:
                return x, end_points

        feature_map = x
        x = F.spatial_avg(x)

        with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
            x = F.conv2d(x, num_classes, (1, 1), use_biases=True, name='Logits')
            heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), use_biases=True, name='Logits')
            end_points['HeatMapFeatures'] = heat_map_features

        x = F.squeeze_hw(x)
        end_points['Logits'] = x

        return x, end_points


def resNeXt(kind: str, F: NNFunction, layer_counts, num_classes, x, use_se=False, num_groups=32, base_width=4,
            dropout_rate=None, initial_conv_channels=32, keep_in_out_channels_same_conv=True):
    assert F.dtype == x.dtype, 'model dtype {} != input dtype {}'.format(F.dtype, x.dtype)

    end_points = {}
    with tf.variable_scope('ResNeXt', custom_getter=float32_variable_storage_getter):
        with tf.variable_scope('InitConv'):
            if kind in ['A', 'B']:
                x = F.conv2d_bn_relu(x, initial_conv_channels, [7, 7], [2, 2], padding='SAME_RESNET',
                                     name='conv2d_bn_relu1')
            else:
                x = F.conv2d_bn_relu(x, initial_conv_channels, [3, 3], [2, 2], padding='SAME_RESNET',
                                     name='conv2d_bn_relu1_1')
                x = F.conv2d_bn_relu(x, initial_conv_channels, [3, 3], name='conv2d_bn_relu1_2')
                x = F.conv2d_bn_relu(x, initial_conv_channels * 2, [3, 3], name='conv2d_bn_relu1_3')

        x = F.max_pool2d(x, [3, 3], [2, 2], name='MaxPool')

        bottleneck = bottleneck_map[kind]
        for i, count in enumerate(layer_counts):
            resnet_in_channels = 64 * (2 ** i)
            group_width = base_width * resnet_in_channels / 64
            bottleneck_channels = group_width * num_groups
            output_channels = resnet_in_channels * 4

            with tf.variable_scope('Layer{}'.format(i + 1)):
                for j in range(count):
                    with tf.variable_scope('Block{}'.format(j + 1)):
                        x = bottleneck(F, x, output_channels, bottleneck_channels,
                                       stride=2 if j == 0 and i != 0 else 1,
                                       downsample=True if j == 0 else False, use_se=use_se,
                                       num_groups=num_groups,
                                       keep_in_out_channels_same_conv=keep_in_out_channels_same_conv)

        feature_map = x
        x = F.spatial_avg(x)

        if dropout_rate is not None:
            x = F.dropout(x, dropout_rate, name='Dropout')
        with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):
            x = F.conv2d(x, num_classes, (1, 1), use_biases=True, name='Logits')
            heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), use_biases=True, name='Logits')

        x = F.squeeze_hw(x)
        end_points['Logits'] = x
        end_points['HeatMapFeatures'] = heat_map_features

        return x, end_points


choices = {
    50: [3, 4, 6, 3],
    101: [3, 4, 23, 3],
    152: [3, 8, 36, 3],
    200: [3, 24, 36, 3]
}


def resNeXt_50_32x4d_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resNeXt('D', F, layer_counts, num_classes, x, num_groups=32, base_width=4)


def resNeXt_50_64x4d_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resNeXt('D', F, layer_counts, num_classes, x, num_groups=64, base_width=4)


def resNeXt_101_64x4d_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[101]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resNeXt('D', F, layer_counts, num_classes, x, num_groups=64, base_width=4)


def SENet_154(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[152]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resNeXt('D', F, layer_counts, num_classes, x, num_groups=64, base_width=4, use_se=True,
                             dropout_rate=0.2)


# resnet A

return_layer_map = {
    2: 'Layer_0',
    3: 'Layer_1',
    4: 'Layer_2',
    5: 'Layer_3',
}


def resnet_v1_backbone(F: NNFunction, num_layers=50, kind='A', min_level=2, max_level=5):
    layer_counts = choices[num_layers]

    return_layer = return_layer_map[max_level]

    def fn(x):
        _, end_points = resnet_v1(kind, F, layer_counts, None, x, return_layer=return_layer)
        features_map = {}
        for i in range(min_level, max_level + 1):
            features_map[i] = end_points[return_layer_map[i]]

        return features_map

    return fn


def resnet_v1_50(num_classes, is_training, dtype=tf.float32, data_format='NCHW', return_layer=None):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('A', F, layer_counts, num_classes, x, return_layer=return_layer)


def resnet_v1_50_drop_block(num_classes, is_training, dtype=tf.float32, data_format='NCHW', return_layer=None, drop_block_keep_rate=0.9):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('A', F, layer_counts, num_classes, x, return_layer=return_layer, drop_block_keep_rate=drop_block_keep_rate)


def resnet_v1_50_merge(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('A', F, layer_counts, num_classes, x, keep_in_out_channels_same_conv=False)


def resnet_v1_101(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[101]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('A', F, layer_counts, num_classes, x)


def resnet_v1_152(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[152]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('A', F, layer_counts, num_classes, x)


def resnet_v1_200(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[200]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('A', F, layer_counts, num_classes, x)


# resnet B

def resnet_v1_50_B(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('B', F, layer_counts, num_classes, x)


def resnet_v1_101_B(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[101]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('B', F, layer_counts, num_classes, x)


def resnet_v1_152_B(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[152]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('B', F, layer_counts, num_classes, x)


def resnet_v1_200_B(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[200]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('B', F, layer_counts, num_classes, x)


# resent C

def resnet_v1_50_C(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('C', F, layer_counts, num_classes, x)


def resnet_v1_101_C(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[101]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('C', F, layer_counts, num_classes, x)


def resnet_v1_152_C(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[152]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('C', F, layer_counts, num_classes, x)


def resnet_v1_200_C(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[200]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('C', F, layer_counts, num_classes, x)


# resent D

def se_resnet_v1_50_D(num_classes, is_training, dtype=tf.float32, data_format='NHWC', return_layer=None):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('D', F, layer_counts, num_classes, x, use_se=True, return_layer=return_layer)


def linear_resnet_v1_50_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('L', F, layer_counts, num_classes, x)


def resnet_v1_50_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW', return_layer=None):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('D', F, layer_counts, num_classes, x, return_layer=return_layer)


def resnet_v1_50_D_drop_block(num_classes, is_training, dtype=tf.float32, data_format='NCHW', return_layer=None, drop_block_keep_rate=0.9):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('D', F, layer_counts, num_classes, x, return_layer=return_layer, drop_block_keep_rate=drop_block_keep_rate)


def resnet_v1_50_D_merge(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[50]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('D', F, layer_counts, num_classes, x, keep_in_out_channels_same_conv=False)


def resnet_v1_101_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW', return_layer=None):
    layer_counts = choices[101]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('D', F, layer_counts, num_classes, x, return_layer=return_layer)


def resnet_v1_152_D(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    layer_counts = choices[152]
    F = NNFunction(is_training, dtype, data_format)
    return lambda x: resnet_v1('D', F, layer_counts, num_classes, x)


if __name__ == '__main__':
    def sv():
        for i in tf.global_variables():
            print(i)


    tf.reset_default_graph()
    F = NNFunction(data_format='NHWC')
    model = resnet_v1_backbone(F, min_level=3, max_level=5)

    img = tf.ones([1, 640, 640, 3], dtype=F.dtype)
    net_points = model(img)
    print(net_points)

    from models import show_model

    show_model()
