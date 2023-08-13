from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Conv2D, Conv1D, BatchNormalization, Activation, GlobalAveragePooling2D, Lambda, \
    Reshape, AlphaDropout
import tensorflow as tf
import numpy as np

from ghost_bottleneck.bottleneck import GBNeck
from ghost_bottleneck.mycnn import Conv0

# tf.config.experimental_run_functions_eagerly(True)

from typing import Callable, Optional
from typing import Any, Optional, Sequence, Tuple
from leaf_audio import convolution
from leaf_audio import initializers
from leaf_audio import pooling
from leaf_audio import postprocessing
import tensorflow_addons as tfa

_TensorCallable = Callable[[tf.Tensor], tf.Tensor]
_Initializer = tf.keras.initializers.Initializer


class SquaredModulus(tf.keras.layers.Layer):
    def __init__(self):
        super().__init__(name='squared_modulus')
        self._pool = tf.keras.layers.AveragePooling1D(pool_size=2, strides=2)

    def call(self, x):
        x = tf.transpose(x, perm=[0, 2, 1])
        output = 2 * self._pool(x ** 2)
        return tf.transpose(output, perm=[0, 2, 1])


class GhostNet(Model):
    """
    The main GhostNet architecture as specified in "GhostNet: More Features from Cheap Operations"
    Paper:
    https://arxiv.org/pdf/1911.11907.pdf

    """

    def __init__(self, classes,
                 conv1d_cls=convolution.GaborConv1D,
                 activation=SquaredModulus(),
                 pooling_cls=pooling.GaussianLowpass,
                 # n_filters: int = 40,
                 n_filters: int = 64,
                 sample_rate: int = 44100,
                 window_len: float = 25.,
                 window_stride: float = 10.,
                 preemp_init: _Initializer = initializers.PreempInit(),
                 complex_conv_init: _Initializer = initializers.GaborInit(
                     sample_rate=44100, min_freq=60.0, max_freq=22050),
                 pooling_init: _Initializer = tf.keras.initializers.Constant(0.4),
                 regularizer_fn: Optional[tf.keras.regularizers.Regularizer] = None,
                 compression_fn: _TensorCallable = postprocessing.PCENLayer(
                     alpha=0.96,
                     smooth_coef=0.1,
                     delta=2.0,
                     floor=1e-12,
                     trainable=True,
                     learn_smooth_coef=True,
                     per_channel_smooth_coef=True),
                 preemp: bool = True,
                 mean_var_norm: bool = True,
                 ):
        super(GhostNet, self).__init__()  # 对父类属性初始化
        self.classes = classes

        # 下面是leaf卷积
        window_size = int(sample_rate * window_len // 1000 + 1)
        window_stride = int(sample_rate * window_stride // 1000)
        if preemp:
            self._preemp_conv = tf.keras.layers.Conv1D(filters=1,
                                                       kernel_size=2,
                                                       strides=1,
                                                       padding='SAME',
                                                       use_bias=False,
                                                       input_shape=(None, None, 1),
                                                       kernel_initializer=preemp_init,
                                                       kernel_regularizer=regularizer_fn,
                                                       name='tfbanks_preemp',
                                                       trainable=True)
        self._complex_conv = conv1d_cls(filters=2 * n_filters,
                                        kernel_size=window_size,
                                        strides=1,
                                        padding='SAME',
                                        use_bias=False,
                                        input_shape=(None, None, 1),
                                        kernel_initializer=complex_conv_init,
                                        kernel_regularizer=regularizer_fn,
                                        name='tfbanks_complex_conv',
                                        trainable=True)
        self._activation = activation
        self._pooling = pooling_cls(kernel_size=window_size,
                                    strides=window_stride,
                                    padding='SAME',
                                    use_bias=False,
                                    kernel_initializer=pooling_init,
                                    kernel_regularizer=regularizer_fn,
                                    trainable=True)
        self._instance_norm = None
        if mean_var_norm:
            self._instance_norm = tfa.layers.InstanceNormalization(axis=2,
                                                                   epsilon=1e-6,
                                                                   center=True,
                                                                   scale=True,
                                                                   beta_initializer='zeros',
                                                                   gamma_initializer='ones',
                                                                   name='tfbanks_instancenorm')
        self._compress_fn = compression_fn if compression_fn else tf.identity
        # self._spec_augment_fn = postprocessing.SpecAugment(
        # ) if spec_augment else tf.identity
        self._spec_augment_fn = tf.identity
        self._preemp = preemp

        # 下面是原始ghostnet网络
        self.conv1 = Conv2D(16, (3, 3), strides=(2, 2), padding='same',
                            activation=None, use_bias=False)
        self.conv2 = Conv2D(960, (1, 1), strides=(1, 1), padding='same', data_format='channels_last',
                            activation=None, use_bias=False)
        self.conv3 = Conv2D(1280, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.conv4 = Conv2D(self.classes, (1, 1), strides=(1, 1), padding='same',
                            activation=None, use_bias=False)
        self.drop = AlphaDropout(0.5)
        for i in range(3):
            setattr(self, f"batchnorm{i + 1}", BatchNormalization())
        self.relu = Activation('relu')
        self.softmax = Activation('softmax')
        self.squeeze = Lambda(self._squeeze)
        self.reshape = Lambda(self._reshape)
        self.pooling = GlobalAveragePooling2D()

        self.dwkernels = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
        self.strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]
        self.exps = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
        self.outs = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]

        # self.dwkernels = [3, 3, 5, 5, 5, 5, 5, 5, 5, 5]
        # self.strides = [2, 2, 1, 2, 1, 1, 1, 2, 1, 1]
        # self.exps = [16, 72, 88, 96, 240, 120, 144, 288, 576, 576]
        # self.outs = [16, 24, 24, 40, 40, 48, 48, 96, 96, 96]

        self.ratios = [2] * 16
        self.use_ses = [False, False, False, True, True, False, False, False,
                        False, True, True, True, False, True, False, True]
        # self.use_ses = [True, False, False, True, True, True, True, True, True, True]
        for i, args in enumerate(zip(self.dwkernels, self.strides, self.exps, self.outs, self.ratios, self.use_ses)):
            setattr(self, f"gbneck{i}", GBNeck(*args))

    @staticmethod
    def _squeeze(x):
        """
        移除尺寸为 1 的所有轴
        """
        return K.squeeze(x, 1)

    @staticmethod
    def _reshape(x):
        return Reshape((1, 1, int(x.shape[1])))(x)

    def call(self, inputs):
        # outputs = inputs[:, :, tf.newaxis] if inputs.shape.ndims < 3 else inputs
        #leaf
        outputs = tf.expand_dims(inputs, axis=-1) #扩维
        outputs = self._preemp_conv(outputs)   #预加重
        outputs = self._complex_conv(outputs)
        outputs = self._activation(outputs)    #激活
        outputs = self._pooling(outputs)      #池化
        outputs = tf.maximum(outputs, 1e-5)
        outputs = self._compress_fn(outputs)
        if self._instance_norm is not None:
            outputs = self._instance_norm(outputs)
        # outputs = self._spec_augment_fn(outputs)
        x = tf.expand_dims(outputs, axis=-1)

        # 下面是ghostnet网络
        x = self.relu(self.batchnorm1(self.conv1(x)))
        # Iterate through Ghost Bottlenecks
        for i in range(12):
            x = getattr(self, f"gbneck{i}")(x)
        x = self.relu(self.batchnorm2(self.conv2(x)))
        x = self.reshape(self.pooling(x))
        x = self.drop(x)
        x = self.relu(self.batchnorm3(self.conv3(x)))
        x = self.drop(x)
        x = self.conv4(x)
        # x = self.drop(x)
        x = self.squeeze(x)
        output = self.softmax(x)
        # print(output.shape)
        return output

    def summary_model(self):
        inputs = tf.keras.Input(shape=(64000,))  #15360,64000,176400
        outputs = self.call(inputs)
        tf.keras.Model(inputs=inputs, outputs=outputs, name="thing").summary()


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
      tensor: A tensor of any type.
    Returns:
      A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape


def convolutional_block_attention_module(feature_map, inner_units_ratio=0.5):
    """
    CBAM: convolution block attention module, which is described in "CBAM: Convolutional Block Attention Module"
    Architecture : "https://arxiv.org/pdf/1807.06521.pdf"
    If you want to use this module, just plug this module into your network
    :param feature_map : input feature map
    :param index : the index of convolution block attention module
    :param inner_units_ratio: output units number of fully connected layer: inner_units_ratio*feature_map_channel
    :return:feature map with channel and spatial attention
    """
    feature_map_shape = [32, 30, 40, 5]
    # channel attention
    channel_avg_weights = tf.nn.avg_pool(
        input=feature_map,
        ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )
    channel_max_weights = tf.nn.max_pool(
        input=feature_map,
        ksize=[1, feature_map_shape[1], feature_map_shape[2], 1],
        strides=[1, 1, 1, 1],
        padding='VALID'
    )
    channel_avg_reshape = tf.reshape(channel_avg_weights,
                                     [feature_map_shape[0], 1, feature_map_shape[3]])
    channel_max_reshape = tf.reshape(channel_max_weights,
                                     [feature_map_shape[0], 1, feature_map_shape[3]])
    channel_w_reshape = tf.concat([channel_avg_reshape, channel_max_reshape], axis=1)

    fc_1 = tf.keras.layers.Dense(
        units=feature_map_shape[3] * inner_units_ratio,
        activation=tf.nn.relu
    )(channel_w_reshape)
    fc_2 = tf.keras.layers.Dense(
        units=feature_map_shape[3],
        activation=None
    )(fc_1)
    channel_attention = tf.reduce_sum(fc_2, axis=1, name="channel_attention_sum")
    channel_attention = tf.nn.sigmoid(channel_attention, name="channel_attention_sum_sigmoid")
    channel_attention = tf.reshape(channel_attention, shape=[feature_map_shape[0], 1, 1, feature_map_shape[3]])
    feature_map_with_channel_attention = tf.multiply(feature_map, channel_attention)
    # spatial attention
    channel_wise_avg_pooling = tf.reduce_mean(feature_map_with_channel_attention, axis=3)
    channel_wise_max_pooling = tf.reduce_max(feature_map_with_channel_attention, axis=3)

    channel_wise_avg_pooling = tf.reshape(channel_wise_avg_pooling,
                                          shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                 1])
    channel_wise_max_pooling = tf.reshape(channel_wise_max_pooling,
                                          shape=[feature_map_shape[0], feature_map_shape[1], feature_map_shape[2],
                                                 1])

    channel_wise_pooling = tf.concat([channel_wise_avg_pooling, channel_wise_max_pooling], axis=3)
    spatial_attention = Conv2D(
        1,
        [3, 3],
        padding='SAME',
        activation=tf.nn.sigmoid
    )(channel_wise_pooling)
    feature_map_with_attention = tf.multiply(feature_map_with_channel_attention, spatial_attention)
    return feature_map_with_attention
