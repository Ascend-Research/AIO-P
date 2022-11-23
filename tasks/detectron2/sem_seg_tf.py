import tensorflow.compat.v1 as tf
from tasks.detectron2.mask_rcnn_tf import TFMaskRCNN

class TFPanSeg(tf.keras.Model):
    def __init__(self):
        super(TFPanSeg, self).__init__()
        self.inst_seg_module = TFMaskRCNN()
        self.sem_seg_module = TFSemSeg()

    def call(self, x, training=True):
        return self.inst_seg_module(x, training=training) + [self.sem_seg_module(x, training=training)]

class TFSemSeg(tf.keras.Model):
    def __init__(self):
        super(TFSemSeg, self).__init__()

        self.sem_seg_head = TFSemSegHead()

    def call(self, x, training=True):
        return self.sem_seg_head(x, training=training)


class TFSemSegHead(tf.keras.Model):
    def __init__(self):
        super(TFSemSegHead, self).__init__()
        self.heads = []

        # p2
        p2 = tf.keras.Sequential()
        p2.add(
            tf.keras.layers.Conv2D(
                filters=128, kernel_size=3, strides=1, padding="same", use_bias=False
            )
        )
        p2.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        self.heads.append(p2)

        # p3
        p3 = tf.keras.Sequential()
        p3.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=2, strides=2, activation="relu"
            )
        )
        p3.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        self.heads.append(p3)

        # p4
        p4 = tf.keras.Sequential()
        p4.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=2, strides=2, activation="relu"
            )
        )
        p4.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        p4.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=2, strides=2, activation="relu"
            )
        )
        p4.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        self.heads.append(p4)

        # p5
        p5 = tf.keras.Sequential()
        p5.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=2, strides=2, activation="relu"
            )
        )
        p5.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        p5.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=2, strides=2, activation="relu"
            )
        )
        p5.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        p5.add(
            tf.keras.layers.Conv2DTranspose(
                filters=128, kernel_size=2, strides=2, activation="relu"
            )
        )
        p5.add(
            tf.keras.layers.BatchNormalization(axis=3, trainable=True, momentum=0.1)
        )
        self.heads.append(p5)

        # Predictor
        self.predictor = tf.keras.layers.Conv2D(filters=54, kernel_size=1, strides=1)

    def call(self, x, training=True):
        x = x[1:]
        x = x[::-1]
        x_sum = None
        for i, head in enumerate(self.heads):
            if i == 0:
                x_sum = head(x[i], training=training)
            else:
                x_sum = x_sum + head(x[i], training=training)
        return self.predictor(x_sum)
