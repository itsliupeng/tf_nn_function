import tensorflow as tf

from models.nn_functions import F_scope
from models.nn_functions import NNFunction, float32_variable_storage_getter


def MBConvBlock(F: NNFunction, x, o, k, s, e, se_ratio=None, name=None, relu_fn=tf.nn.swish):
    with F_scope(name):
        identity = x
        i = F.get_channels(x)

        if e != 1:
            x = F.conv2d_bn(x, i * e, (1, 1), name='Conv2dS1_BN')
            x = relu_fn(x)

        x = F.depthwise_conv2d_bn(x, 1, (k, k), (s, s), name='DwConv2dS{}_BN'.format(s))
        x = relu_fn(x)

        if se_ratio is not None:
            num_reduced_filters = max(1, int(i * se_ratio))
            x = F.se(x, num_reduced_filters=num_reduced_filters, name='SE')

        x = F.conv2d_bn(x, o, (1, 1), name='Conv2dS1_BN_Linear')

        if s == 1 and F.get_channels(identity) == F.get_channels(x):
            x += identity

        return x


# repeat r, filter_size k, stride s,  expansion ratio e, input channels i, output channels o, se ratio
MBConv_blocks_config = [
    [1, 3, 1, 1, 32, 16, 0.25],
    [2, 3, 2, 6, 16, 24, 0.25],
    [2, 5, 2, 6, 24, 40, 0.25],
    [3, 3, 2, 6, 40, 80, 0.25],
    [3, 5, 1, 6, 80, 112, 0.25],
    [4, 5, 2, 6, 112, 192, 0.25],
    [1, 3, 1, 6, 192, 320, 0.25]
]


def efficient_net_b0(num_classes, is_training, dtype=tf.float32, data_format='NCHW', relu_fn=tf.nn.swish):
    F = NNFunction(is_training, dtype, data_format)

    def fn(x):
        end_points = {}
        with tf.variable_scope('EfficientNet', custom_getter=float32_variable_storage_getter):
            with tf.variable_scope('InitConv'):
                x = F.conv2d_bn(x, 32, (3, 3), (2, 2), name='Conv2dS2_BN')
                x = relu_fn(x)

            for i, (r, k, s, e, _, o, se_ratio) in enumerate(MBConv_blocks_config):
                with tf.variable_scope('Layer_{}'.format(i)):
                    for j in range(r):
                        x = MBConvBlock(F, x, o, k, s=s if j == 0 else 1, e=e, se_ratio=se_ratio, name='MBConv_{}'.format(j))

            if num_classes > 320:
                with tf.variable_scope('LastConv'):
                    x = F.conv2d_bn(x, 1280, (1, 1), name='Conv2dS1_BN')
                    x = relu_fn(x)

            feature_map = x
            x = F.spatial_avg(x)

            x = F.dropout(x, rate=0.2)

            with tf.variable_scope('FC', reuse=tf.AUTO_REUSE):
                x = F.conv2d(x, num_classes, (1, 1), use_biases=True, name='Logits')
                heat_map_features = F.conv2d(feature_map, num_classes, (1, 1), use_biases=True, name='Logits')
                end_points['HeatMapFeatures'] = heat_map_features

            x = F.squeeze_hw(x)
            end_points['Logits'] = x

        return x, end_points

    return fn


if __name__ == '__main__':

    tf.reset_default_graph()

    model = efficient_net_b0(1001, is_training=True, dtype=tf.float16, data_format='NHWC')
    img = tf.placeholder(tf.float16, [1, 224, 224, 3])
    labels = tf.placeholder(tf.int32, [1, 1])
    _logits, _end_points = model(img)
    loss = tf.losses.sparse_softmax_cross_entropy(labels, _logits)

    grad_vars = tf.gradients(loss, tf.trainable_variables())


    def exclude_batch_norm_and_bias(name):
        for i in ['gamma', 'beta', 'batch_normalization', 'BatchNorm']:
            if i in name:
                return False
        return True


    trainable_variables_without_bn = [v for v in tf.trainable_variables() if exclude_batch_norm_and_bias(v.name)]
    for i in trainable_variables_without_bn:
        print(i)

    from models import show_model

    show_model()
    # calculate_flops()
