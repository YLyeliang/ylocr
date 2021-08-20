import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
from ..core.conv_utils import convBlock, bottleNeck, ResLayer


class ResNet(keras.Model):
    arch_settings = {
        # 18: (BasicBlock, (2, 2, 2, 2)),
        # 34: (BasicBlock, (3, 4, 6, 3)),
        50: (bottleNeck, (3, 4, 6, 3)),
        101: (bottleNeck, (3, 4, 23, 3)),
        152: (bottleNeck, (3, 8, 36, 3))

    }

    def __init__(self, depth=50,
                 stem_channels=64,
                 strides=((1, 1), (2, 2), (2, 1), (2, 1)),
                 dilations=(1, 1, 1, 1),
                 act='relu'):
        super(ResNet, self).__init__()

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks

        self.make_stem(stem_channels)
        self.inplanes = stem_channels

        self.res_layer = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            planes = stem_channels * 2 ** i  # 64 128 256 512
            res_layer = ResLayer(self.block, planes, kernel_size=3, strides=stride, act=act, num_blocks=num_blocks,
                                 name=f"conv{i + 1}")
            self.res_layer.append(res_layer)


    def make_stem(self, stem_channels):
        self.stem = keras.Sequential([layers.Conv2D(stem_channels, kernel_size=3, kernel_initializer='he_normal',
                                                    padding='SAME', activation='relu', name='conv1'),
                                      convBlock(stem_channels, 3, (2, 2), act='relu', name="conv2")
                                      ])

    def call(self, x, training=None, mask=None):
        x = self.stem(x)

        for i, res_layer in enumerate(self.res_layer):
            x = res_layer(x)
        return x

    def build(self):
        self._is_graph_network = True
        self._init_graph_net_work(
            inputs=self.input_layer,
            outputs=self.out)


class ResNetV2(ResNet):
    def __init__(self,
                 depth=50,
                 stem_channels=64,
                 strides=((1, 1), (2, 2), (2, 1), (2, 1)),
                 dilations=(1, 1, 1, 1),
                 act='relu',
                 deep_stem=True):
        super(ResNetV2, self).__init__(depth, stem_channels, strides, dilations, act=act, )

    def call(self, x, training=None, mask=None):
        x = self.stem(x)

        for i, res_layer in enumerate(self.res_layer):
            x = res_layer(x)
        return x
