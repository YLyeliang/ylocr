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
                 out_indices=(0, 1, 2, 3),
                 act='relu',
                 deep_stem=False):
        super(ResNet, self).__init__()

        self.deep_stem = deep_stem

        self.block, stage_blocks = self.arch_settings[depth]
        self.stage_blocks = stage_blocks

        self.make_stem(stem_channels)
        self.inplanes = stem_channels

        self.res_layer = []

        for i, num_blocks in enumerate(self.stage_blocks):
            stride = strides[i]
            dilation = dilations[i]
            planes = stem_channels * 2 ** i  # 64 128 256 512
            res_layer = ResLayer(self.block, planes, kernel_size=3, strides=stride, act=act, num_blocks=num_blocks,
                                 name=f"conv{i + 1}")
            self.res_layer.append(res_layer)

    def make_stem(self, stem_channels):
        if self.deep_stem:
            self.stem = keras.Sequential([convBlock(stem_channels, 7, 2, 'relu', name="stem"),
                                          ])
        else:
            self.conv1 = convBlock(stem_channels, 7, 2, 'relu', name="stem")
        self.maxpool = layers.MaxPool2D((3, 3), 2, padding='SAME', name="pool")

    def call(self, x, training=None, mask=None):
        if self.deep_stem:
            x = self.stem(x)
        else:
            x = self.conv1(x)
        x = self.maxpool(x)

        for i, res_layer in enumerate(self.res_layer):
            x = res_layer(x)
        return x
