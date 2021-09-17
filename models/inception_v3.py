from functools import partial

import tensorflow as tf

from models.nn_functions import NNFunction, float32_variable_storage_getter


def inception_v3_base(F: NNFunction, inputs, min_depth=16, depth_multiplier=1.0, max_reduction_idx=None, use_se=False, name=None):
    """Inception model from http://arxiv.org/abs/1512.00567.

    Constructs an Inception v3 network from inputs to the given final endpoint.
    This method can construct the network up to the final inception block
    Mixed_7c.

    Note that the names of the layers in the paper do not correspond to the names
    of the endpoints registered by this function although they build the same
    network.

    Here is a mapping from the old_names to the new names:
    Old name          | New name
    =======================================
    conv0             | Conv2d_1a_3x3
    conv1             | Conv2d_2a_3x3
    conv2             | Conv2d_2b_3x3
    pool1             | MaxPool_3a_3x3
    conv3             | Conv2d_3b_1x1
    conv4             | Conv2d_4a_3x3
    pool2             | MaxPool_5a_3x3
    mixed_35x35x256a  | Mixed_5b
    mixed_35x35x288a  | Mixed_5c
    mixed_35x35x288b  | Mixed_5d
    mixed_17x17x768a  | Mixed_6a
    mixed_17x17x768b  | Mixed_6b
    mixed_17x17x768c  | Mixed_6c
    mixed_17x17x768d  | Mixed_6d
    mixed_17x17x768e  | Mixed_6e
    mixed_8x8x1280a   | Mixed_7a
    mixed_8x8x2048a   | Mixed_7b
    mixed_8x8x2048b   | Mixed_7c

    Args:
      F: NNFunction
      inputs: a tensor of size [batch_size, height, width, channels].
      final_endpoint: specifies the endpoint to construct the network up to. It
        can be one of ['Conv2d_1a_3x3', 'Conv2d_2a_3x3', 'Conv2d_2b_3x3',
        'MaxPool_3a_3x3', 'Conv2d_3b_1x1', 'Conv2d_4a_3x3', 'MaxPool_5a_3x3',
        'Mixed_5b', 'Mixed_5c', 'Mixed_5d', 'Mixed_6a', 'Mixed_6b', 'Mixed_6c',
        'Mixed_6d', 'Mixed_6e', 'Mixed_7a', 'Mixed_7b', 'Mixed_7c'].
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      name: Optional variable_scope.

    Returns:
      tensor_out: output tensor corresponding to the final_endpoint.
      end_points: a set of activations for external use, for example summaries or
                  losses.

    Raises
      ValueError: if final_endpoint is not set to one of the predefined values,
                  or depth_multiplier <= 0
    """
    # end_points will collect relevant activations for external use, for example
    # summaries or losses.
    end_points = {}

    if depth_multiplier <= 0:
        raise ValueError('depth_multiplier is not greater than zero.')
    depth = lambda d: max(int(d * depth_multiplier), min_depth)

    with tf.variable_scope(name, 'InceptionV3'):
        # with slim.arg_scope([F.conv2d_bn_relu, slim.max_pool2d, slim.avg_pool2d],
        #                     stride=1, padding='VALID'):
        # 299 x 299 x 3
        net = F.conv2d_bn_relu(inputs, depth(32), [3, 3], [2, 2], name='Conv2d_1a_3x3', padding='VALID')
        # 149 x 149 x 32
        net = F.conv2d_bn_relu(net, depth(32), [3, 3], name='Conv2d_2a_3x3', padding='VALID')
        # 147 x 147 x 32
        net = F.conv2d_bn_relu(net, depth(64), [3, 3], padding='SAME', name='Conv2d_2b_3x3')
        # 147 x 147 x 64
        net = F.max_pool2d(net, [3, 3], [2, 2], padding='VALID', name='MaxPool_3a_3x3')
        # 73 x 73 x 64
        net = F.conv2d_bn_relu(net, depth(80), [1, 1], padding='VALID', name='Conv2d_3b_1x1')
        # 73 x 73 x 80.
        net = F.conv2d_bn_relu(net, depth(192), [3, 3], padding='VALID', name='Conv2d_4a_3x3')
        # 71 x 71 x 192.
        if max_reduction_idx is not None:
            end_points['reduction_2'] = net
            if max_reduction_idx == 2:
                return x, end_points
        net = F.max_pool2d(net, [3, 3], [2, 2], padding='VALID', name='MaxPool_5a_3x3')
        # 35 x 35 x 192.

    concat_axis = F.channel_axis

    # mixed: 35 x 35 x 256.
    with tf.variable_scope('Mixed_5b'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(48), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(64), [5, 5], name='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(96), [3, 3], name='Conv2d_0b_3x3')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(96), [3, 3], name='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(32), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

    # mixed_1: 35 x 35 x 288.
    with tf.variable_scope('Mixed_5c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(48), [1, 1], name='Conv2d_0b_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(64), [5, 5], name='Conv_1_0c_5x5')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(96), [3, 3], name='Conv2d_0b_3x3')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(96), [3, 3], name='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], [1, 1], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(64), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

        # mixed_2: 35 x 35 x 288.
    with tf.variable_scope('Mixed_5d'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(48), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(64), [5, 5], name='Conv2d_0b_5x5')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(96), [3, 3], name='Conv2d_0b_3x3')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(96), [3, 3], name='Conv2d_0c_3x3')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(64), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

    if max_reduction_idx is not None:
        end_points['reduction_3'] = net
        if max_reduction_idx == 3:
            return x, end_points

    # mixed_3: 17 x 17 x 768.
    with tf.variable_scope('Mixed_6a'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(384), [3, 3], [2, 2], padding='VALID', name='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(64), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(96), [3, 3], name='Conv2d_0b_3x3')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(96), [3, 3], [2, 2], padding='VALID', name='Conv2d_1a_1x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.max_pool2d(net, [3, 3], [2, 2], padding='VALID', name='MaxPool_1a_3x3')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2])
        if use_se:
            net = F.se(net, name='SE')

    # mixed4: 17 x 17 x 768.
    with tf.variable_scope('Mixed_6b'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(128), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(128), [1, 7], name='Conv2d_0b_1x7')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [7, 1], name='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(128), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(128), [7, 1], name='Conv2d_0b_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(128), [1, 7], name='Conv2d_0c_1x7')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(128), [7, 1], name='Conv2d_0d_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [1, 7], name='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(192), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

    # mixed_5: 17 x 17 x 768.
    with tf.variable_scope('Mixed_6c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(160), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(160), [1, 7], name='Conv2d_0b_1x7')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [7, 1], name='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(160), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(160), [7, 1], name='Conv2d_0b_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(160), [1, 7], name='Conv2d_0c_1x7')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(160), [7, 1], name='Conv2d_0d_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [1, 7], name='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(192), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

    # mixed_6: 17 x 17 x 768.
    with tf.variable_scope('Mixed_6d'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(160), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(160), [1, 7], name='Conv2d_0b_1x7')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [7, 1], name='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(160), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(160), [7, 1], name='Conv2d_0b_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(160), [1, 7], name='Conv2d_0c_1x7')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(160), [7, 1], name='Conv2d_0d_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [1, 7], name='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(192), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

    # mixed_7: 17 x 17 x 768.
    with tf.variable_scope('Mixed_6e'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [1, 7], name='Conv2d_0b_1x7')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [7, 1], name='Conv2d_0c_7x1')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [7, 1], name='Conv2d_0b_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [1, 7], name='Conv2d_0c_1x7')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [7, 1], name='Conv2d_0d_7x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(192), [1, 7], name='Conv2d_0e_1x7')
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(192), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        end_points['Mixed_6e'] = net
        if use_se:
            net = F.se(net, name='SE')

    if max_reduction_idx is not None:
        end_points['reduction_4'] = net
        if max_reduction_idx == 4:
            return x, end_points

    # mixed_8: 8 x 8 x 1280.
    with tf.variable_scope('Mixed_7a'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
            branch_0 = F.conv2d_bn_relu(branch_0, depth(320), [3, 3], [2, 2], padding='VALID', name='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(192), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [1, 7], name='Conv2d_0b_1x7')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [7, 1], name='Conv2d_0c_7x1')
            branch_1 = F.conv2d_bn_relu(branch_1, depth(192), [3, 3], [2, 2], padding='VALID', name='Conv2d_1a_3x3')
        with tf.variable_scope('Branch_2'):
            branch_2 = F.max_pool2d(net, [3, 3], [2, 2], padding='VALID', name='MaxPool_1a_3x3')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2])
        if use_se:
            net = F.se(net, name='SE')

    # mixed_9: 8 x 8 x 2048.
    with tf.variable_scope('Mixed_7b'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(320), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(384), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = tf.concat(axis=concat_axis, values=[
                F.conv2d_bn_relu(branch_1, depth(384), [1, 3], name='Conv2d_0b_1x3'),
                F.conv2d_bn_relu(branch_1, depth(384), [3, 1], name='Conv2d_0b_3x1')])
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(448), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(384), [3, 3], name='Conv2d_0b_3x3')
            branch_2 = tf.concat(axis=concat_axis, values=[
                F.conv2d_bn_relu(branch_2, depth(384), [1, 3], name='Conv2d_0c_1x3'),
                F.conv2d_bn_relu(branch_2, depth(384), [3, 1], name='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(192), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        if use_se:
            net = F.se(net, name='SE')

    # mixed_10: 8 x 8 x 2048.
    with tf.variable_scope('Mixed_7c'):
        with tf.variable_scope('Branch_0'):
            branch_0 = F.conv2d_bn_relu(net, depth(320), [1, 1], name='Conv2d_0a_1x1')
        with tf.variable_scope('Branch_1'):
            branch_1 = F.conv2d_bn_relu(net, depth(384), [1, 1], name='Conv2d_0a_1x1')
            branch_1 = tf.concat(axis=concat_axis, values=[
                F.conv2d_bn_relu(branch_1, depth(384), [1, 3], name='Conv2d_0b_1x3'),
                F.conv2d_bn_relu(branch_1, depth(384), [3, 1], name='Conv2d_0c_3x1')])
        with tf.variable_scope('Branch_2'):
            branch_2 = F.conv2d_bn_relu(net, depth(448), [1, 1], name='Conv2d_0a_1x1')
            branch_2 = F.conv2d_bn_relu(branch_2, depth(384), [3, 3], name='Conv2d_0b_3x3')
            branch_2 = tf.concat(axis=concat_axis, values=[
                F.conv2d_bn_relu(branch_2, depth(384), [1, 3], name='Conv2d_0c_1x3'),
                F.conv2d_bn_relu(branch_2, depth(384), [3, 1], name='Conv2d_0d_3x1')])
        with tf.variable_scope('Branch_3'):
            branch_3 = F.avg_pool2d(net, [3, 3], name='AvgPool_0a_3x3')
            branch_3 = F.conv2d_bn_relu(branch_3, depth(192), [1, 1], name='Conv2d_0b_1x1')
        net = tf.concat(axis=concat_axis, values=[branch_0, branch_1, branch_2, branch_3])
        end_points['Mixed_7c'] = net
        if use_se:
            net = F.se(net, name='SE')

    if max_reduction_idx is not None:
        end_points['reduction_5'] = net
        if max_reduction_idx == 5:
            return x, end_points
    
    end_points['reduction_5'] = net

    return net, end_points


def _reduced_kernel_size_for_small_input(F: NNFunction, input_tensor, kernel_size):
    """Define kernel size which is automatically reduced for small input.

    If the shape of the input images is unknown at graph construction time this
    function assumes that the input images are is large enough.

    Args:
      input_tensor: input tensor of size [batch_size, height, width, channels].
      kernel_size: desired kernel size of length 2: [kernel_height, kernel_width]

    Returns:
      a tensor with the kernel size.

    TODO(jrru): Make this function work with unknown shapes. Theoretically, this
    can be done with the code below. Problems are two-fold: (1) If the shape was
    known, it will be lost. (2) inception.slim_raw.ops._two_element_tuple cannot
    handle tensors that define the kernel size.
        shape = tf.shape(input_tensor)
        return = tf.stack([tf.minimum(shape[1], kernel_size[0]),
                           tf.minimum(shape[2], kernel_size[1])])

    """

    if F.channel_axis == -1:
        shape = input_tensor.get_shape().as_list()
        h, w = shape[1], shape[2]
    else:
        shape = input_tensor.get_shape().as_list()
        h, w = shape[2], shape[3]
    if h is None or w is None:
        kernel_size_out = kernel_size
    else:
        kernel_size_out = [min(h, kernel_size[0]),
                           min(w, kernel_size[1])]
    return kernel_size_out


def get_inception_v3_fn(num_classes, is_training, dtype=tf.float16, data_format='NCHW', use_se=False, name=None):
    """Inception model from http://arxiv.org/abs/1512.00567.

    "Rethinking the Inception Architecture for Computer Vision"

    Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens,
    Zbigniew Wojna.

    With the default arguments this method constructs the exact model defined in
    the paper. However, one can experiment with variations of the inception_v3
    network by changing arguments dropout_keep_prob, min_depth and
    depth_multiplier.

    The default image size used to train this network is 299x299.

    Args:
      inputs: a tensor of size [batch_size, height, width, channels].
      num_classes: number of predicted classes. If 0 or None, the logits layer
        is omitted and the input features to the logits layer (before dropout)
        are returned instead.
      is_training: whether is training or not.
      dropout_keep_prob: the percentage of activation values that are retained.
      min_depth: Minimum depth value (number of channels) for all convolution ops.
        Enforced when depth_multiplier < 1, and not an active constraint when
        depth_multiplier >= 1.
      depth_multiplier: Float multiplier for the depth (number of channels)
        for all convolution ops. The value must be greater than zero. Typical
        usage will be to set this value in (0, 1) to reduce the number of
        parameters or computation cost of the model.
      spatial_squeeze: if True, logits is of shape [B, C], if false logits is of
          shape [B, 1, 1, C], where B is batch_size and C is number of classes.
      reuse: whether or not the network and its variables should be reused. To be
        able to reuse 'scope' must be given.
      create_aux_logits: Whether to create the auxiliary logits.
      scope: Optional variable_scope.
      global_pool: Optional boolean flag to control the avgpooling before the
        logits layer. If false or unset, pooling is done with a fixed window
        that reduces default-sized inputs to 1x1, while larger inputs lead to
        larger outputs. If true, any input size is pooled down to 1x1.

    Returns:
      net: a Tensor with the logits (pre-softmax activations) if num_classes
        is a non-zero integer, or the non-dropped-out input to the logits layer
        if num_classes is 0 or None.
      end_points: a dictionary from components of the network to the corresponding
        activation.

    Raises:
      ValueError: if 'depth_multiplier' is less than or equal to zero.
    """

    F = NNFunction(is_training, dtype, data_format)
    dropout_rate = 0.2
    min_depth = 16
    depth_multiplier = 1.0

    def inception_v3_fn(inputs):
        if depth_multiplier <= 0:
            raise ValueError('depth_multiplier is not greater than zero.')
        depth = lambda d: max(int(d * depth_multiplier), min_depth)

        with tf.variable_scope(name, 'InceptionV3', custom_getter=float32_variable_storage_getter):
            net, end_points = inception_v3_base(F, inputs, min_depth=min_depth, depth_multiplier=depth_multiplier,
                                                use_se=use_se, name=name)

            # Auxiliary
            aux_logits = end_points['Mixed_6e']
            with tf.variable_scope('AuxLogits'):
                aux_logits = F.avg_pool2d(aux_logits, [5, 5], [3, 3], padding='VALID', name='AvgPool_1a_5x5')
                aux_logits = F.conv2d_bn_relu(aux_logits, depth(128), [1, 1], name='Conv2d_1b_1x1')

                # Shape of feature map before the final layer.
                kernel_size = _reduced_kernel_size_for_small_input(F, aux_logits, [5, 5])
                aux_logits = F.conv2d_bn_relu(aux_logits, depth(768), kernel_size, padding='VALID',
                                              kernel_initializer=tf.truncated_normal_initializer(stddev=0.01),
                                              name='Conv2d_2a_{}x{}'.format(*kernel_size))
                aux_logits = F.conv2d(aux_logits, num_classes, [1, 1], use_biases=True,
                                      kernel_initializer=tf.truncated_normal_initializer(stddev=0.001),
                                      name='Conv2d_2b_1x1')
                aux_logits = F.squeeze_hw(aux_logits, name='SpatialSqueeze')
                end_points['AuxLogits'] = aux_logits

            # Logits
            with tf.variable_scope('Logits'):
                # 8 x 8 x 2048
                feature_map = net

                # Pooling with a fixed kernel size.
                kernel_size = _reduced_kernel_size_for_small_input(F, net, [8, 8])
                net = F.avg_pool2d(net, kernel_size, padding='VALID', name='AvgPool_1a_{}x{}'.format(*kernel_size))

                # 1 x 1 x 2048
                net = F.dropout(net, rate=dropout_rate, name='Dropout_1b')

                end_points['Embedding'] = F.squeeze_hw(net)

                # 2048
                with tf.variable_scope("FC", reuse=tf.AUTO_REUSE):
                    logits = F.conv2d(net, num_classes, [1, 1], use_biases=True, name='Conv2d_1c_1x1')
                    heat_map_features = F.conv2d(feature_map, num_classes, [1, 1], use_biases=True,
                                                 name='Conv2d_1c_1x1')

                logits = F.squeeze_hw(logits, name='SpatialSqueeze')
                end_points['Logits'] = logits
                end_points['HeatMapFeatures'] = heat_map_features

        return logits, end_points

    return inception_v3_fn


inception_v3 = partial(get_inception_v3_fn, use_se=False)
se_inception_v3 = partial(get_inception_v3_fn, use_se=True)


def inception_v3_backbone(F: NNFunction, use_se=False, min_level=2, max_level=5):
    def fn(x):
        _, end_points = inception_v3_base(F, x, max_reduction_idx=max_level, use_se=use_se, name='InceptionV3')
        feature_maps = {}
        for i in range(min_level, max_level + 1):
            feature_maps[i] = end_points['reduction_{}'.format(i)]

        return feature_maps

    return fn


if __name__ == '__main__':
    dtype = tf.float16

    F = NNFunction(data_format='NHWC', dtype=dtype, is_training=True)
    x = tf.random_normal([1, 299, 299, 3], dtype=dtype)
    model = inception_v3_backbone(F, min_level=2, max_level=5)
    feature_maps = model(x)

    from models import show_model

    show_model()
    print('Done')
