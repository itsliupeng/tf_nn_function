# NN_FUNCTION，build tensorflow model as pytorch nn functional style

tf.layers or slim is too sophisticated to build models efficiently. They all call tf.nn in low level.

We provide `NNFunciton` wrapping tf.nn functions to simplify building neural network like pytorch, just claiming ` F = NNFunction(is_training, dtype, data_format)`.

## base model
We evaluate most image classification models using horovod + tf.estimator + tf.data, it stays state of art at most cases.

resnet and resNeXt models use preprocessing of RESNET (resnet ppr. as short) as default. (resize min 255,then crop 224 x 224)

(inference GPU: 1080Ti, without any optimization)

| name                | STA   | top1   | top5   | inference speed(fp32) | inference speed(fp16) | loss  | others                             |
| ------------------- | ----- | ------ | ------ | --------------------- | --------------------- | ----- | ---------------------------------- |
| inception_v3        | 78.8  | 79.248 | 94.432 | 3.661                 | 2.688                 | 1.653 | 299,ls,e300                        |
| ,224x224            |       | 77.158 | 93.496 | 2.251                 |                       | 1.818 | 224,e120,inception ppr.            |
| se_inception_v3     |       | 79.938 | 94.678 | 3.817                 | 2.6400                | 1.661 | 299,ls,e200                        |
| ,e120               |       | 79.472 | 94.656 | 3.788                 | 2.708                 | 1.719 |                                    |
| resnet_v1_50        |       | 76.174 | 92.988 | 2.276                 |                       | 1.411 |                                    |
| mobilenet_v1_1.0    | 72.93 | 73.23  | 91.264 | 1.321                 |                       | 1.382 | ls,e200                            |
| mobilenet_v2_1.4    | 75.0  | 74.536 | 91.966 | 2.111                 |                       | 1.391 | ls,e120                            |
| ,dw w/o relu6       |       | 74.928 | 92.192 | 2.049                 |                       | 1.358 | ls,e200                            |
| mobilenet_v2_1.0    |       | 71.506 | 90.26  | 1.637                 |                       | 1.518 | ls,e120                            |
| ,dw w/o relu6       | 71.8  | 71.6   | 90.242 | 1.337                 |                       | 1.491 | ls,e200                            |
| resnet_v1_101_D     | 79.78 | 79.572 | 94.818 | 3.483                 | 2.935                 | 1.229 | resnet ppr                         |
| ,320x320            |       | 80.804 | 95.414 | 8.642                 |                       | 1.147 | 320,ls,e120,inception ppr.         |
| resNeXt_50_32x4d_D  |       | 78.192 | 94.014 | 3.173                 | 2.681                 | 1.265 |                                    |
| resNeXt_50_64x4d_D  |       | 78.348 | 94.206 | 3.899                 |                       | 1.383 |                                    |
| resNeXt_101_64x4d_D |       | 79.432 | 94.6   | 6.396                 | 5.417                 | 1.342 |                                    |
| ,320x320            |       | 80.304 | 95.24  | 13.988                |                       | 1.302 | 320,ls,e120,fp16, bs32             |
| resnet_v1_50_D      |       | 78.342 | 94.16  | 2.520                 | 2.117                 | 1.251 | e120,resnet ppr.                   |
| ,inceptio ppr.      | 78.48 | 78.444 | 94.26  | 2.803                 |                       | 1.246 | e120,inception ppr.                |
| ,320x320            |       | 78.984 | 94.664 | 5.399                 |                       | 1.217 | e120, inception ppr. fp16          |
| ,se                 |       | 79.192 | 94.678 | 2.998                 |                       | 1.199 | e200, inception bbox ppr. fp32, ls |
| shufflenet_v2_2.0   | 74.9  | 74.848 | 92.146 | 1.772                 | 1.577                 | 1.334 | e200, inception bbox ppr, fp32, ls |
| ,epoch120           |       | 74.018 | 91.738 |                       | 1.567                 | 1.393 | inception bbox ppr, fp32, ls       |