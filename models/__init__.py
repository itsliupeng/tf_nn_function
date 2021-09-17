import os

import tensorflow as tf

from models.efficientnet import efficient_net_b0
from models.inception_v3 import inception_v3, se_inception_v3
from models.m_nasnet import MnasNet_A1, MnasNet_small, MnasNet_d1_320
from models.mobilenet_v1 import mobilenet_v1_1_0
from models.mobilenet_v2 import mobilenet_v2_1_4, mobilenet_v2_1_0, mobilenet_v2_0_75, mobilenet_v2_0_5, \
    mobilenet_v2_0_25, mobilenet_v2_1_0_merge, mobilenet_v2_1_4_merge
from models.nets.mobilenet.mobilenet_v2 import mobilenet_v2_050 as slim_mobilenet_v2_050
from models.resnet import resnet_v1_50, resnet_v1_101, resnet_v1_50_B, resnet_v1_50_C, resnet_v1_50_D, \
    se_resnet_v1_50_D, linear_resnet_v1_50_D, resnet_v1_101_D, resNeXt_50_32x4d_D, resNeXt_101_64x4d_D, \
    resNeXt_50_64x4d_D, SENet_154, resnet_v1_50_D_merge, resnet_v1_50_merge, resnet_v1_50_drop_block, resnet_v1_50_D_drop_block
from models.shufflenet_v2 import shufflenet_v2_2_0, shufflenet_v2_50, se_shufflenet_v2_164, se_shufflenet_v2_2_0, \
    se_shufflenet_v2_50, shufflenet_v2_1_0, shufflenet_v2_0_5, shufflenet_v2_2_0_B, shufflenet_v2_1_0_B, \
    shufflenet_v2_2_0_C, shufflenet_v2_1_0_C, res_shufflenet_v2_2_0_C

model_map = {
    'inception_v3': inception_v3,
    'se_inception_v3': se_inception_v3,
    'resnet_v1_50': resnet_v1_50,
    'resnet_v1_101': resnet_v1_101,
    'resnet_v1_50_drop_block': resnet_v1_50_drop_block,
    'resnet_v1_50_B': resnet_v1_50_B,
    'resnet_v1_50_C': resnet_v1_50_C,
    'resnet_v1_50_D': resnet_v1_50_D,
    'resnet_v1_50_D_drop_block': resnet_v1_50_D_drop_block,
    'resnet_v1_101_D': resnet_v1_101_D,
    'resnet_v1_50_merge': resnet_v1_50_merge,
    'resnet_v1_50_D_merge': resnet_v1_50_D_merge,
    'linear_resnet_v1_50_D': linear_resnet_v1_50_D,
    'se_resnet_v1_50_D': se_resnet_v1_50_D,
    'mobilenet_v1_1_0': mobilenet_v1_1_0,
    'mobilenet_v2_1.4': mobilenet_v2_1_4,
    'mobilenet_v2_1.0': mobilenet_v2_1_0,
    'mobilenet_v2_0.75': mobilenet_v2_0_75,
    'mobilenet_v2_0.5': mobilenet_v2_0_5,
    'mobilenet_v2_0.25': mobilenet_v2_0_25,
    'mobilenet_v2_1_0_merge': mobilenet_v2_1_0_merge,
    'mobilenet_v2_1_4_merge': mobilenet_v2_1_4_merge,
    'shufflenet_v2_0_5': shufflenet_v2_0_5,
    'shufflenet_v2_1_0': shufflenet_v2_1_0,
    'shufflenet_v2_2_0': shufflenet_v2_2_0,
    'shufflenet_v2_50': shufflenet_v2_50,
    'se_shufflenet_v2_50': se_shufflenet_v2_50,
    'se_shufflenet_v2_164': se_shufflenet_v2_164,
    'se_shufflenet_v2_2_0': se_shufflenet_v2_2_0,
    'shufflenet_v2_2_0_B': shufflenet_v2_2_0_B,
    'shufflenet_v2_1_0_B': shufflenet_v2_1_0_B,
    'shufflenet_v2_2_0_C': shufflenet_v2_2_0_C,
    'shufflenet_v2_1_0_C': shufflenet_v2_1_0_C,
    'res_shufflenet_v2_2_0_C': res_shufflenet_v2_2_0_C,
    'resNeXt_50_32x4d_D': resNeXt_50_32x4d_D,
    'resNeXt_50_64x4d_D': resNeXt_50_64x4d_D,
    'resNeXt_101_64x4d_D': resNeXt_101_64x4d_D,
    'SENet_154': SENet_154,
    'MnasNet_A1': MnasNet_A1,
    'MnasNet_small': MnasNet_small,
    'MnasNet_d1_320': MnasNet_d1_320,
    'efficient_net_b0': efficient_net_b0,
}

MODEL_BASE_DIR = '/share/JeanWe/models'
pretrained_model_variable_exclusions_out_shape_map = {
    'inception_v3': [os.path.join(MODEL_BASE_DIR, 'inception_v3_top1_0.7876/model.ckpt-75068'),
                     ['InceptionV3/Logits', 'InceptionV3/AuxLogits/Conv2d_2b_1x1'],
                     (299, 299, 3)],
    'resnet_v1_50': [os.path.join(MODEL_BASE_DIR, 'resnet_v1_50_top1_0.7714/model.ckpt-37534'),
                     ['ResNet/FC'],
                     (224, 224, 3)],
    'resnet_v1_50_drop_block': [None, [], None],
    'resnet_v1_50_D': [os.path.join(MODEL_BASE_DIR, 'resnet_v1_50_D_top1_0.7844/model.ckpt-37534'),
                       ['ResNet/FC'],
                       (224, 224, 3)],
    'resnet_v1_101_D': [os.path.join(MODEL_BASE_DIR, 'resent_v1_101_D_top1_0.7946/model.ckpt-75068'),
                        ['ResNet/FC'],
                        (224, 224, 3)],
    'resnet_v1_50_D_drop_block': [None, [], None],
}


def get_model(model_type, num_classes, is_training, dtype=tf.float16, data_format='NCHW'):
    if model_type.startswith('slim_'):
        model_type = model_type[5:]
        assert dtype == tf.float32, 'slim model dtype should be tf.float32'
        assert data_format == 'NHWC', 'slim model data_format should be NHWC'
        from models.nets.nets_factory import get_network_fn
        return get_network_fn(model_type, num_classes=num_classes, is_training=is_training)

    else:
        assert model_type in model_map.keys()
        return model_map[model_type](num_classes, is_training, dtype, data_format)


def show_model(show_all_vars=False):
    prev = None
    for var in tf.global_variables():
        if var.name.split('/')[-1] in ['beta:0', 'moving_mean:0', 'moving_variance:0']:
            continue

        if prev is None:
            print('{} - {}'.format(var.name, var.shape.as_list()))
        else:
            idx = idx_a_minus_b(var.name, prev.name)
            short_name = var.name[idx:]
            if short_name.startswith('/bn'):
                short_name = '/bn'
            print('{}{} - {}'.format(' ' * idx, short_name, var.shape.as_list()))
        prev = var

    if show_all_vars:
        from jeanwe.utils import pretty_list_str
        print(pretty_list_str(tf.global_variables()))


def idx_a_minus_b(a, b):
    a_splits = a.split('/')
    b_splits = b.split('/')
    for i in range(min(len(a_splits), len(b_splits))):
        if a_splits[i] != b_splits[i]:
            break

    return len('/'.join(a_splits[0:i]))


def calculate_flops():
    # Print to stdout an analysis of the number of floating point operations in the
    # model broken down by individual operations.
    tf.profiler.profile(tf.get_default_graph(), options=tf.profiler.ProfileOptionBuilder.float_operation(), cmd='scope')


def show_parameters():
    tf.profiler.profile(tf.get_default_graph(),
                        options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter(), cmd='scope')

# if __name__ == '__main__':
# tf.reset_default_graph()
# model = se_resNext_v1_50_D(1001, is_training=True, dtype=tf.float16, data_format='NHWC')
#
# img = tf.ones([1, 299, 299, 3], dtype=tf.float16)
# logits_, net_points = model(img)
#
# show_model()
