import copy
import tensorflow.compat.v1 as tf
from model_src.search_space.ofa_profile.constants import *
from model_src.search_space.ofa_profile.operations_tf import *
from model_src.search_space.ofa_profile.arch_utils import get_final_channel_sizes


class OFAProxylessStem(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=3, stride=2, padding="same", name="OFAProxylessStem",
                 data_format="channels_last", use_bias=False):
        super(OFAProxylessStem, self).__init__()
        self._name = name
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                           padding=padding, data_format=data_format,
                                           activation=None, use_bias=use_bias,
                                           kernel_initializer="he_normal", name=self._name)
        self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis,
                                                     momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                     name=self._name + "/bn")
        self.act = tf.keras.layers.ReLU(max_value=6)

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            x = self.conv(x)
            x = self.bn(x, training=training)
            x = self.act(x)
        return x


class OFAMbv3Stem(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=3, stride=2, padding="same", name="OFAMbv3Stem",
                 data_format="channels_last", use_bias=False):
        super(OFAMbv3Stem, self).__init__()
        self._name = name
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=stride,
                                           padding=padding, data_format=data_format,
                                           activation=None, use_bias=use_bias,
                                           kernel_initializer="he_normal", name=self._name)
        self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis,
                                                     momentum=BN_MOMENTUM, epsilon=1e-5,
                                                     name=self._name + "/bn")
        self.act = HSwish()

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            x = self.conv(x)
            x = self.bn(x, training=training)
            x = self.act.call(x)
        return x


class OFAProxylessOutLayer(tf.keras.layers.Layer):

    def __init__(self, C_hidden, n_classes,
                 name="OFAProxylessOutLayer",
                 data_format="channels_last",
                 deconv_skip=False):
        super(OFAProxylessOutLayer, self).__init__()
        self._name = name
        self.deconv_skip = deconv_skip
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv = tf.keras.layers.Conv2D(filters=C_hidden, kernel_size=1, strides=1,
                                           padding="same", data_format=data_format,
                                           activation=None, use_bias=False,
                                           kernel_initializer="he_normal",
                                           name=self._name + "/conv")
        self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                     momentum=BN_MOMENTUM, epsilon=BN_EPSILON,
                                                     name=self._name + "/bn")
        self.act = tf.keras.layers.ReLU(max_value=6)
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format)
        self.classifier = tf.keras.layers.Dense(n_classes, name=self._name + "/fc")

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            x = self.act(self.bn(self.conv(x), training=training))
            if self.deconv_skip:
                logits = x
            else:
                out = self.global_pooling(x)
                out= self.dropout(out, training=training)
                logits = self.classifier(out)
        return logits


class OFAMbv3OutLayer(tf.keras.layers.Layer):

    def __init__(self, hidden_1, hidden_2, n_classes,
                 name="OFAMbv3OutLayer",
                 data_format="channels_last",
                 deconv_skip=False):
        super(OFAMbv3OutLayer, self).__init__()
        self._name = name
        self.deconv_skip = deconv_skip
        if data_format == "channels_last":
            self.channel_axis = 3
        else:
            self.channel_axis = 1
        self.conv1 = tf.keras.layers.Conv2D(filters=hidden_1, kernel_size=1, strides=1,
                                            padding="same", data_format=data_format,
                                            activation=None, use_bias=False,
                                            kernel_initializer="he_normal",
                                            name=self._name + "/conv1")
        self.bn = tf.keras.layers.BatchNormalization(axis=self.channel_axis, trainable=True,
                                                     momentum=BN_MOMENTUM, epsilon=1e-5,
                                                     name=self._name + "/bn")
        self.act = HSwish()
        self.conv2 = tf.keras.layers.Conv2D(filters=hidden_2, kernel_size=1, strides=1,
                                            padding="same", data_format=data_format,
                                            activation=None, use_bias=False,
                                            kernel_initializer="he_normal",
                                            name=self._name + "/conv2")
        self.dropout = tf.keras.layers.Dropout(0.1)
        self.global_pooling = tf.keras.layers.GlobalAveragePooling2D(data_format)
        self.classifier = tf.keras.layers.Dense(n_classes, name=self._name + "/fc")

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            x = self.act.call(self.bn(self.conv1(x), training=training))
            if self.deconv_skip:
                logits = x
            else:
                x = self.global_pooling(x)
                x = tf.reshape(x, (x.shape[0], 1, 1, x.shape[1]))
                out = self.act.call(self.conv2(x))
                out = tf.reshape(out, (out.shape[0], out.shape[-1]))
                out= self.dropout(out, training=training)
                logits = self.classifier(out)
        return logits


class OFAProxylessNet(tf.keras.Model):

    def __init__(self, net_configs,  # A list of block name lists, group by stages
                 C_init=32, n_classes=1000,
                 strides=(2, 2, 2, 1, 2, 1),
                 block_channel_sizes=(16, 24, 40, 80, 96, 192, 320),
                 out_hidden_size=1280,
                 name="OFAProxylessNet",
                 data_format="channels_last", w=OFA_W_PN, skip_list=None):
        super(OFAProxylessNet, self).__init__()
        assert len(net_configs) == len(strides) == len(block_channel_sizes) - 1
        C_init = get_final_channel_size(C_init, w)
        block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
        out_hidden_size = get_final_channel_size(out_hidden_size, w)
        self.w = w
        self._name = name
        self.data_format = data_format
        self.net_configs = net_configs
        self.strides = strides
        self.block_channel_sizes = block_channel_sizes
        self.skip_list = skip_list
        if self.skip_list is not None:
            self.call = self.call_skip
        else:
            self.call = self.call_class

        self.stem = OFAProxylessStem(C_init, data_format=self.data_format)
        first_block = MBConv2(C_init, block_channel_sizes[0], 3, 1,
                              name="FirstB", data_format=self.data_format,
                              expansion_ratio=1, enable_skip=False)
        self.blocks = [first_block]
        C_in = block_channel_sizes[0]
        for i, b_ops in enumerate(net_configs):
            b_stride = strides[i]
            C_out = self.block_channel_sizes[i + 1]
            for op_i, op_name in enumerate(b_ops):
                block = get_tf_op(op_name=op_name, C_in=C_in, C_out=C_out,
                                  stride=b_stride if op_i == 0 else 1,
                                  scope_name="Block_{}_{}".format(i, op_i),
                                  data_format=self.data_format)
                self.blocks.append(block)
                C_in = C_out
        self.out_net = OFAProxylessOutLayer(out_hidden_size, n_classes,
                                            deconv_skip=True if self.skip_list is not None else False)

    def call_class(self, inputs, training=True):
        with tf.name_scope(self._name):
            x = self.stem(inputs, training=training)
            for bi, block in enumerate(self.blocks):
                x = block.call(x, training=training)
            x = self.out_net(x, training=training)
        return x

    def call_skip(self, inputs, training=True):
        x_list = []
        with tf.name_scope(self._name):
            x = self.stem(inputs, training=training)
            for bi, block in enumerate(self.blocks):
                x = block.call(x, training=training)
                if bi in self.skip_list:
                    x_list.append(x)
            x = self.out_net(x, training=training)
            x_list.append(x)
        return x_list


class OFAMbv3Net(tf.keras.Model):

    def __init__(self, net_configs,  # A list of block name lists, group by stages
                 C_init=16, n_classes=1000,
                 strides=(2, 2, 2, 1, 2),
                 block_channel_sizes=(16, 24, 40, 80, 112, 160),
                 stage_act=("relu", "relu", "swish", "swish", "swish"),
                 stage_se_ratios=(0., 0.25, 0., 0.25, 0.25),
                 name="OFAMbv3Net", hidden_1=960, hidden_2=1280,
                 data_format="channels_last", w=OFA_W_MBV3, skip_list=None):
        super(OFAMbv3Net, self).__init__()
        assert len(net_configs) == len(strides) == len(block_channel_sizes) - 1
        C_init = get_final_channel_size(C_init, w)
        block_channel_sizes = get_final_channel_sizes(block_channel_sizes, w)
        hidden_1 = get_final_channel_size(hidden_1, w)
        hidden_2 = get_final_channel_size(hidden_2, w)
        self.w = w
        self._name = name
        self.data_format = data_format
        self.net_configs = net_configs
        self.strides = strides
        self.block_channel_sizes = block_channel_sizes
        self.skip_list = skip_list
        if self.skip_list is not None:
            self.call = self.call_skip
        else:
            self.call = self.call_class

        self.stem = OFAMbv3Stem(C_init, data_format=self.data_format)
        first_block = MBConv3(C_init, block_channel_sizes[0], 3, 1,
                              name="FirstB", data_format=self.data_format,
                              expansion_ratio=1, enable_skip=True,
                              se_ratio=0., act_type="relu")
        self.blocks = [first_block]
        C_in = block_channel_sizes[0]
        for i, b_ops in enumerate(net_configs):
            b_stride = strides[i]
            act_type = stage_act[i]
            se_ratio = stage_se_ratios[i]
            C_out = self.block_channel_sizes[i + 1]
            for op_i, op_name in enumerate(b_ops):
                block = get_tf_op(op_name=op_name, C_in=C_in, C_out=C_out,
                                  stride=b_stride if op_i == 0 else 1,
                                  scope_name="Block_{}_{}".format(i, op_i),
                                  data_format=self.data_format,
                                  act_type=act_type, se_ratio=se_ratio)
                self.blocks.append(block)
                C_in = C_out
        self.out_net = OFAMbv3OutLayer(hidden_1, hidden_2, n_classes,
                                       data_format=self.data_format,
                                       deconv_skip=True if self.skip_list is not None else False)

    def call_class(self, inputs, training=True):
        with tf.name_scope(self._name):
            x = self.stem.call(inputs, training=training)
            for bi, block in enumerate(self.blocks):
                x = block.call(x, training=training)
            x = self.out_net.call(x, training=training)
        return x

    def call_skip(self, inputs, training=True):
        x_list = []
        with tf.name_scope(self._name):
            x = self.stem.call(inputs, training=training)
            for bi, block in enumerate(self.blocks):
                x = block.call(x, training=training)
                if bi in self.skip_list:
                    x_list.append(x)
            x = self.out_net.call(x, training=training)
            x_list.append(x)
        return x_list


class OFAResNetStem(tf.keras.Model):

    def __init__(self, C_mid, C_out,
                 input_stem_skipping,
                 name="OFAResNetStem",
                 data_format="channels_last"):
        super(OFAResNetStem, self).__init__()
        self._name = name
        self.data_format = data_format
        self.input_stem_skipping = input_stem_skipping
        if input_stem_skipping:
            self.conv1 = AugmentedConv2D(filters=C_mid,
                                         kernel_size=3, stride=2, padding="same",
                                         activation=tf.keras.layers.ReLU(),
                                         data_format=data_format,
                                         use_bn=True, use_bias=False,
                                         name=self._name + "/stem/conv1")
            self.conv2 = None
            self.conv3 = AugmentedConv2D(filters=C_out,
                                         kernel_size=3, stride=1, padding="same",
                                         activation=tf.keras.layers.ReLU(),
                                         data_format=data_format,
                                         use_bn=True, use_bias=False,
                                         name=self._name + "/stem/conv3")
        else:
            self.conv1 = AugmentedConv2D(filters=C_mid,
                                         kernel_size=3, stride=2, padding="same",
                                         activation=tf.keras.layers.ReLU(),
                                         data_format=data_format,
                                         use_bn=True, use_bias=False,
                                         name=self._name + "/stem/conv1")
            self.conv2 = ResidualBlock(
                AugmentedConv2D(filters=C_mid,
                                kernel_size=3, stride=1, padding="same",
                                activation=tf.keras.layers.ReLU(),
                                data_format=data_format,
                                use_bn=True, use_bias=False,
                                name=self._name + "/stem/conv2"),
                Identity(name=self._name + "/stem/conv2_res"),
            )
            self.conv3 = AugmentedConv2D(filters=C_out,
                                         kernel_size=3, stride=1, padding="same",
                                         activation=tf.keras.layers.ReLU(),
                                         data_format=data_format,
                                         use_bn=True, use_bias=False,
                                         name=self._name + "/stem/conv3")

    def call(self, x, training=True):
        x = self.conv1(x, training=training)
        if self.conv2 is not None:
            x = self.conv2(x, training=training)
        x = self.conv3(x, training=training)
        return x


class OFAResNet(tf.keras.Model):

    def __init__(self, net_configs,  # A dict containing d, e, w lists
                 width_multipliers=OFA_RES_WIDTH_MULTIPLIERS,
                 stage_min_block_counts=OFA_RES_STAGE_MIN_N_BLOCKS,
                 stage_max_block_counts=OFA_RES_STAGE_MAX_N_BLOCKS,
                 stage_strides=(1, 2, 2, 2),
                 stage_base_channels=OFA_RES_STAGE_BASE_CHANNELS,
                 n_classes=1000, name="OFAResNet",
                 data_format="channels_last", skip_list=None):
        super(OFAResNet, self).__init__()
        self._name = name
        self.data_format = data_format
        self.n_classes = n_classes
        self.stage_strides = stage_strides
        self.stage_min_block_counts = stage_min_block_counts
        self.stage_max_block_counts = stage_max_block_counts
        self.width_multipliers = width_multipliers
        self.stage_base_channels = stage_base_channels
        self.width_mul_inds = net_configs["w"]
        self.depth_opts = net_configs["d"]
        self.exp_ratio_opts = net_configs["e"]
        self.skip_list = skip_list
        if self.skip_list is not None:
            self.call = self.call_skip
        else:
            self.call = self.call_class

        # Stem building
        self.stem_setting = self.depth_opts[0]
        self.input_stem_skipping = self.stem_setting != 2
        self.stem_output_channels = [
            make_divisible(64 * wm, 8) for wm in self.width_multipliers
        ]
        self.stem_mid_channels = [
            make_divisible(channel // 2, 8) for channel in self.stem_output_channels
        ]
        self.stem_mid_channel_idx = self.width_mul_inds[0]
        self.stem_output_channel_idx = self.width_mul_inds[1]
        self.C_stem_mid = self.stem_mid_channels[self.stem_mid_channel_idx]
        self.C_stem_output = self.stem_output_channels[self.stem_output_channel_idx]
        self.stem = OFAResNetStem(self.C_stem_mid, self.C_stem_output,
                                  self.input_stem_skipping,
                                  name=self._name + "/stem",
                                  data_format=self.data_format)

        # Blocks building
        self.blocks = []
        self.stage_width_options = []
        for c in self.stage_base_channels:
            self.stage_width_options.append([
                make_divisible(c * wm, 8) for wm in self.width_multipliers
            ])
        self.stage_block_counts = [
            base_depth + d for base_depth, d in  zip(self.stage_min_block_counts, self.depth_opts[1:])
        ]
        self.stage_width_mul_inds = self.width_mul_inds[2:]
        self.active_e_list = []
        e_list = copy.deepcopy(self.exp_ratio_opts)
        for si, n_active_blocks in enumerate(self.stage_block_counts):
            assert n_active_blocks <= self.stage_max_block_counts[si]
            active_e_vals = e_list[:n_active_blocks]
            e_list = e_list[self.stage_max_block_counts[si]:]
            self.active_e_list.extend(active_e_vals)
        assert len(e_list) == 0, "Invalid e_list input: {}".format(e_list)
        ei = 0
        C_curr = self.C_stem_output
        for si, n_blocks in enumerate(self.stage_block_counts):
            width_idx = self.stage_width_mul_inds[si]
            width = self.stage_width_options[si][width_idx]
            for bi in range(n_blocks):
                stride = self.stage_strides[si] if bi == 0 else 1
                exp_ratio = self.active_e_list[ei]
                block = ResNetBlock(C_in=C_curr, C_out=width,
                                    kernel_size=3, stride=stride,
                                    expansion_ratio=exp_ratio,
                                    downsample_mode="avgpool_conv",
                                    name=self._name + "/stage{}_block{}".format(si, bi),
                                    data_format=self.data_format)
                self.blocks.append(block)
                ei += 1
                C_curr = width
        self.classifier = tf.keras.layers.Dense(n_classes,
                                                use_bias=True,
                                                name=self._name + "/classifier")
        self.global_avg_pool = tf.keras.layers.GlobalAveragePooling2D(name=self._name + "/mean",
                                                                      data_format=self.data_format)
        self.max_pooling = tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding="same",
                                                        name=self._name + "/maxpool",
                                                        data_format=self.data_format)

    def call_class(self, x, training=True):
        with tf.name_scope(self._name):
            x = self.stem(x, training=training)
            x = self.max_pooling(x)
            for block in self.blocks:
                x = block(x, training=training)
            x = self.global_avg_pool(x)
            x = self.classifier(x)
        return x

    def call_skip(self, x, training=True):
        x_list = []
        with tf.name_scope(self._name):
            x = self.stem(x, training=training)
            x = self.max_pooling(x)
            for bi, block in enumerate(self.blocks):
                x = block(x, training=training)
                if bi in self.skip_list:
                    x_list.append(x)
            x_list.append(x)
        return x_list
