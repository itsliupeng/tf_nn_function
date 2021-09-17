# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains a factory for building various models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf

from models.nets import resnet_v2, se_resnet_v1, cifarnet, alexnet, vgg, lenet, overfeat, resnet_v1, mobilenet_v1
from models.nets.mobilenet import mobilenet_v2
from models.nets.nasnet import pnasnet, nasnet

slim = tf.contrib.slim

networks_map = {'alexnet_v2': alexnet.alexnet_v2,
                'cifarnet': cifarnet.cifarnet,
                'overfeat': overfeat.overfeat,
                'vgg_a': vgg.vgg_a,
                'vgg_16': vgg.vgg_16,
                'vgg_19': vgg.vgg_19,
                'lenet': lenet.lenet,
                'resnet_v1_50': resnet_v1.resnet_v1_50,
                'resnet_v1_101': resnet_v1.resnet_v1_101,
                'resnet_v1_152': resnet_v1.resnet_v1_152,
                'resnet_v1_200': resnet_v1.resnet_v1_200,
                'resnet_v2_50': resnet_v2.resnet_v2_50,
                'resnet_v2_101': resnet_v2.resnet_v2_101,
                'resnet_v2_152': resnet_v2.resnet_v2_152,
                'resnet_v2_200': resnet_v2.resnet_v2_200,
                'mobilenet_v1': mobilenet_v1.mobilenet_v1,
                'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_075,
                'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_050,
                'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_025,
                'mobilenet_v2': mobilenet_v2.mobilenet,
                'mobilenet_v2_050': mobilenet_v2.mobilenet_v2_050,
                'mobilenet_v2_140': mobilenet_v2.mobilenet_v2_140,
                'mobilenet_v2_035': mobilenet_v2.mobilenet_v2_035,
                'nasnet_cifar': nasnet.build_nasnet_cifar,
                'nasnet_mobile': nasnet.build_nasnet_mobile,
                'nasnet_large': nasnet.build_nasnet_large,
                'pnasnet_large': pnasnet.build_pnasnet_large,
                'pnasnet_mobile': pnasnet.build_pnasnet_mobile,
                'se_resnet_v1_50': se_resnet_v1.resnet_v1_50,
                'se_resnet_v1_101': se_resnet_v1.resnet_v1_101
                }

arg_scopes_map = {'alexnet_v2': alexnet.alexnet_v2_arg_scope,
                  'cifarnet': cifarnet.cifarnet_arg_scope,
                  'overfeat': overfeat.overfeat_arg_scope,
                  'vgg_a': vgg.vgg_arg_scope,
                  'vgg_16': vgg.vgg_arg_scope,
                  'vgg_19': vgg.vgg_arg_scope,
                  'lenet': lenet.lenet_arg_scope,
                  'resnet_v1_50': resnet_v1.resnet_arg_scope,
                  'resnet_v1_101': resnet_v1.resnet_arg_scope,
                  'resnet_v1_152': resnet_v1.resnet_arg_scope,
                  'resnet_v1_200': resnet_v1.resnet_arg_scope,
                  'resnet_v2_50': resnet_v2.resnet_arg_scope,
                  'resnet_v2_101': resnet_v2.resnet_arg_scope,
                  'resnet_v2_152': resnet_v2.resnet_arg_scope,
                  'resnet_v2_200': resnet_v2.resnet_arg_scope,
                  'mobilenet_v1': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_075': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_050': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v1_025': mobilenet_v1.mobilenet_v1_arg_scope,
                  'mobilenet_v2': mobilenet_v2.training_scope,
                  'mobilenet_v2_050': mobilenet_v2.training_scope,
                  'mobilenet_v2_035': mobilenet_v2.training_scope,
                  'mobilenet_v2_140': mobilenet_v2.training_scope,
                  'nasnet_cifar': nasnet.nasnet_cifar_arg_scope,
                  'nasnet_mobile': nasnet.nasnet_mobile_arg_scope,
                  'nasnet_large': nasnet.nasnet_large_arg_scope,
                  'pnasnet_large': pnasnet.pnasnet_large_arg_scope,
                  'pnasnet_mobile': pnasnet.pnasnet_mobile_arg_scope,
                  'se_resnet_v1_50': se_resnet_v1.resnet_arg_scope,
                  'se_resnet_v1_101': se_resnet_v1.resnet_arg_scope
                  }

exclusion_for_training = {'vgg_16': ['vgg_16/fc8'],
                          'vgg_19': ['vgg_19/fc8'],
                          'inception_v1': ['InceptionV1/Logits'],
                          'inception_v2': ['InceptionV2/Logits'],
                          'inception_branches': ['InceptionV3/Logits', 'InceptionV3/AuxLogits',
                                                 'InceptionV3/ReduceDimension'],
                          'inception_v4': ['InceptionV4/Logits', 'InceptionV4/AuxLogits'],
                          'inception_v5': ['InceptionV3/Logits', 'InceptionV3/AuxLogits',
                                           'InceptionV3/Conv2d_1a_3x3'],
                          'inception_resnet_v2': ['InceptionResnetV2/Logits',
                                                  'InceptionResnetV2/AuxLogits'],
                          'resnet_v1_50': ['resnet_v1_50/logits'],
                          'resnet_v1_101': ['resnet_v1_101/logits'],
                          'resnet_v1_152': ['resnet_v1_152/logits'],
                          'resnet_v2_50': ['resnet_v2_50/logits'],
                          'resnet_v2_101': ['resnet_v2_101/logits'],
                          'resnet_v2_152': ['resnet_v2_152/logits'],
                          'se_resnet_v1_50': ['se_resnet_v1_50/logits'],
                          'mobilenet_v1': ['MobilenetV1/Logits'],
                          'nasnet_large': ['final_layer', 'aux_11'],
                          }


def get_network_fn(name, num_classes=1001, weight_decay=None, is_training=False):
    """Returns a network_fn such as `logits, end_points = network_fn(images)`.

    Args:
      name: The name of the network.
      num_classes: The number of classes to use for classification. If 0 or None,
        the logits layer is omitted and its input features are returned instead.
      weight_decay: The l2 coefficient for the model weights.
      is_training: `True` if the model is being used for training and `False`
        otherwise.

    Returns:
      network_fn: A function that applies the model to a batch of images. It has
        the following signature:
            net, end_points = network_fn(images)
        The `images` input is a tensor of shape [batch_size, height, width, 3]
        with height = width = network_fn.default_image_size. (The permissibility
        and treatment of other sizes depends on the network_fn.)
        The returned `end_points` are a dictionary of intermediate activations.
        The returned `net` is the topmost layer, depending on `num_classes`:
        If `num_classes` was a non-zero integer, `net` is a logits tensor
        of shape [batch_size, num_classes].
        If `num_classes` was 0 or `None`, `net` is a tensor with the input
        to the logits layer of shape [batch_size, 1, 1, num_features] or
        [batch_size, num_features]. Dropout has not been applied to this
        (even if the network's original classification does); it remains for
        the caller to do this or not.

    Raises:
      ValueError: If network `name` is not recognized.
    """
    if name not in networks_map:
        raise ValueError('Name of network unknown %s' % name)
    func = networks_map[name]

    @functools.wraps(func)
    def network_fn(images, **kwargs):
        if weight_decay:
            arg_scope = arg_scopes_map[name](weight_decay=weight_decay)
        else:
            arg_scope = arg_scopes_map[name]()
        with slim.arg_scope(arg_scope):
            return func(images, num_classes, is_training=is_training, **kwargs)

    if hasattr(func, 'default_image_size'):
        network_fn.default_image_size = func.default_image_size

    return network_fn
