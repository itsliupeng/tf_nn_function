import tensorflow as tf

from models.nn_functions import F_scope
from models.nn_functions import NNFunction, float32_variable_storage_getter
from models.utils import round_filters


def MBConv(F: NNFunction, x, o, k, s, e, se_ratio=None, name=None, relu_fn=tf.nn.relu):
    with F_scope(name):
        identity = x
        i = F.get_channels(x)

        if e != 1:
            x = F.conv2d_bn(x, i * e, (1, 1), name='Conv2dS1_BN')
            x = relu_fn(x)

        x = F.depthwise_conv2d_bn_relu6(x, 1, (k, k), (s, s), name='DwConv2dS{}_BN'.format(s))
        x = relu_fn(x)

        if se_ratio is not None:
            num_reduced_filters = max(1, int(i * se_ratio))
            x = F.se(x, num_reduced_filters=num_reduced_filters, name='SE')

        add_skip = False
        if s == 1 and F.get_channels(identity) == o:
            name = 'Conv2dS1_BN_Linear_Add'
            add_skip = True
        else:
            name = 'Conv2dS1_BN_Linear'

        with tf.variable_scope(name):
            x = F.conv2d_bn(x, o, (1, 1))
            if add_skip:
                x += identity

        return x


# repeat r, filter_size k, stride s,  expansion ratio e, input channels i, output channels o
b1_MBConv_blocks_config = {
    'layer': [
        [1, 3, 1, 1, 32, 16, None],
        [3, 3, 2, 3, 16, 24, None],
        [3, 5, 2, 3, 24, 40, None],
        [3, 5, 2, 6, 40, 80, None],
        [2, 3, 1, 6, 80, 96, None],
        [4, 5, 2, 6, 96, 192, None],
        [1, 3, 1, 6, 192, 320, None]
    ],
    'dropout': 0.2}

# mnasnet_a1 = [
#     'r1_k3_s11_e1_i32_o16_noskip',
#     'r2_k3_s22_e6_i16_o24',
#     'r3_k5_s22_e3_i24_o40_se0.25',
#     'r4_k3_s22_e6_i40_o80',
#     'r2_k3_s11_e6_i80_o112_se0.25',
#     'r3_k5_s22_e6_i112_o160_se0.25',
#     'r1_k3_s11_e6_i160_o320'
# ]

a1_MBConv_blocks_config = {
    'layer': [
        [1, 3, 1, 1, 32, 16, None],
        [2, 3, 2, 6, 16, 24, None],
        [3, 5, 2, 3, 24, 40, 0.25],
        [4, 3, 2, 6, 40, 80, None],
        [2, 3, 1, 6, 80, 112, 0.25],
        [3, 5, 2, 6, 112, 160, 0.25],
        [1, 3, 1, 6, 160, 320, None]
    ],
    'dropout': 0.2}

small_MBConv_blocks_config = {
    'layer': [
        [1, 3, 1, 1, 16, 8, None],
        [1, 3, 2, 3, 8, 16, None],
        [2, 3, 2, 6, 16, 16, None],
        [4, 5, 2, 6, 16, 32, 0.25],
        [3, 3, 11, 6, 32, 32, 0.25],
        [3, 5, 2, 6, 32, 88, 0.25],
        [1, 3, 1, 6, 88, 144, None]
    ],
    'dropout': 0.0}

d1_MBConv_blocks_config = {
    'layer': [
        [1, 3, 1, 9, 32, 24, None],
        [3, 3, 2, 9, 24, 36, None],
        [5, 3, 2, 9, 36, 48, None],
        [4, 5, 2, 9, 48, 96, None],
        [5, 7, 1, 3, 96, 96, None],
        [3, 3, 2, 9, 96, 80, None],
        [1, 7, 1, 6, 80, 320, None]
    ],
    'dropout': 0.2}

d1_320_MBConv_blocks_config = {
    'layer': [
        [3, 5, 1, 6, 32, 24, None],
        [4, 7, 2, 9, 24, 36, None],
        [5, 5, 2, 9, 36, 48, None],
        [5, 7, 2, 6, 48, 96, None],
        [5, 3, 1, 9, 96, 144, None],
        [5, 5, 2, 6, 144, 160, None],
        [1, 7, 1, 9, 160, 320, None]],
    'dropout': 0.2}

config_map = {
    'a1': a1_MBConv_blocks_config,
    'b1': b1_MBConv_blocks_config,
    'small': small_MBConv_blocks_config,
    'd1': d1_MBConv_blocks_config,
    'd1_320': d1_320_MBConv_blocks_config,

}


def MnasNetBase(blocks_config, num_classes, is_training, dtype=tf.float32, data_format='NCHW', relu_fn=tf.nn.relu, multiplier=1.0, depth_divisor=8,
                min_depth=None, max_reduction_idx=None):
    F = NNFunction(is_training, dtype, data_format)

    def fn(x):
        end_points = {}
        with tf.variable_scope('MnasNet', custom_getter=float32_variable_storage_getter):
            with tf.variable_scope('InitConv'):
                x = F.conv2d_bn(x, 32, (3, 3), (2, 2), name='Conv2dS2_BN')
                x = relu_fn(x)

            reduction_idx = 1
            for i, (r, k, s, e, _, o, se_ratio) in enumerate(blocks_config['layer']):
                o = round_filters(o, multiplier, depth_divisor, min_depth)

                layer_in = x
                with tf.variable_scope('Layer_{}'.format(i)):
                    for j in range(r):
                        x = MBConv(F, x, o, k, s=s if j == 0 else 1, e=e, name='MBConv_{}'.format(j), se_ratio=se_ratio, relu_fn=relu_fn)

                if max_reduction_idx is not None and s == 2 and reduction_idx <= max_reduction_idx:
                    end_points['reduction_{}'.format(reduction_idx)] = layer_in
                    if reduction_idx == max_reduction_idx:
                        return x, end_points

                    reduction_idx += 1

            if max_reduction_idx is not None and reduction_idx <= max_reduction_idx:
                end_points['reduction_{}'.format(reduction_idx)] = x
                return x, end_points

            if num_classes > 320:
                with tf.variable_scope('LastConv'):
                    x = F.conv2d_bn(x, 1280, (1, 1), name='Conv2dS1_BN')
                    x = relu_fn(x)

            feature_map = x
            x = F.spatial_avg(x)

            x = F.dropout(x, blocks_config['dropout'])

            with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
                x = F.conv2d(x, num_classes, (1, 1), use_biases=True, name='Logits')
                heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), use_biases=True, name='Logits')
                end_points['HeatMapFeatures'] = heat_map_features

            x = F.squeeze_hw(x)
            end_points['Logits'] = x

        return x, end_points

    return fn


def MnasNet_A1(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    return MnasNetBase(config_map['a1'], num_classes, is_training, dtype=dtype, data_format=data_format)


def MnasNet_B1(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    return MnasNetBase(config_map['b1'], num_classes, is_training, dtype=dtype, data_format=data_format)


def MnasNet_small(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    return MnasNetBase(config_map['small'], num_classes, is_training, dtype=dtype, data_format=data_format)


def MnasNet_d1(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    return MnasNetBase(config_map['d1'], num_classes, is_training, dtype=dtype, data_format=data_format)


def MnasNet_d1_320(num_classes, is_training, dtype=tf.float32, data_format='NCHW'):
    return MnasNetBase(config_map['d1_320'], num_classes, is_training, dtype=dtype, data_format=data_format)


def MnasNet_backbone(F: NNFunction, kind='a1', min_level=2, max_level=5, multiplier=1.0):
    def fn(x):
        _, end_points = MnasNetBase(config_map[kind], None, F.is_training, F.dtype, F.data_foramt, max_reduction_idx=max_level, multiplier=multiplier)(x)

        feature_maps = {}
        for i in range(min_level, max_level + 1):
            feature_maps[i] = end_points['reduction_{}'.format(i)]

        return feature_maps

    return fn


if __name__ == '__main__':
    tf.reset_default_graph()

    F = NNFunction(is_training=True, dtype=tf.float16, data_format="NHWC")
    model = MnasNet_backbone(F, kind='d1_320', min_level=2, max_level=5, multiplier=1.0)
    img = tf.placeholder(tf.float16, [1, 320, 320, 3])
    labels = tf.placeholder(tf.int32, [1, 1])
    feature_maps = model(img)

    from models import show_model

    show_model()
    # calculate_flops()

    print('Done')
