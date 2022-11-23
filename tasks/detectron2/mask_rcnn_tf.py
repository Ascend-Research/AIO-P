import tensorflow.compat.v1 as tf
from tasks.detectron2.faster_rcnn_tf import TFROIHeads, TFRPN

class TFMaskRCNN(tf.keras.Model):
    def __init__(self):
        super(TFMaskRCNN, self).__init__()

        self.rpn = TFRPN()
        self.roi_heads = TFROIHeads(output_size=(14, 14))
        self.mask_head = TFMaskRCNNUpsample()

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            rpn_outputs = self.rpn(x)
            return [self.mask_head(self.roi_heads(x, training=training), training=training)] + rpn_outputs


class TFMaskRCNNUpsample(tf.keras.Model):
    def __init__(self, num_classes=80, channels=256, conv_layers=4):
        super(TFMaskRCNNUpsample, self).__init__()
        layers = []
        for _ in range(conv_layers):
            layers.append(tf.keras.layers.Conv2D(
                filters=channels,
                kernel_size=3,
                strides=1,
                padding="same"
            ))
            layers.append(tf.keras.layers.ReLU())
        
        layers.append(tf.keras.layers.Conv2DTranspose(
            filters=channels,
            kernel_size=2,
            strides=2,
            padding="same"
        ))
        layers.append(tf.keras.layers.ReLU())
        layers.append(tf.keras.layers.Conv2D(
            filters=num_classes,
            kernel_size=1,
            strides=1,
            padding="same"
        ))
        self.sequence = tf.keras.Sequential(layers)

    def call(self, x, training=True):
        return self.sequence(x)