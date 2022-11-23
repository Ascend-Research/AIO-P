import tensorflow.compat.v1 as tf


class TFFasterRCNN(tf.keras.Model):
    def __init__(self):
        super(TFFasterRCNN, self).__init__()

        self.rpn = TFRPN()
        self.roi_heads = TFROIHeads()
        self.head = RCNNHead()

    def call(self, x, training=True):
        with tf.name_scope(self._name):
            logits_and_deltas = self.rpn(x, training=training)
            x_roi = self.roi_heads(x, training=training)
            cls_scores, bbox_preds = self.head(x_roi, training=training)

            return [cls_scores, bbox_preds] + logits_and_deltas

class TFRPN(tf.keras.Model):
    def __init__(self, num_anchors=3, box_dim=4):
        super(TFRPN, self).__init__()
        assert type(num_anchors) is int
        assert type(box_dim) is int

        self.rpn_conv = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=1, padding="same", activation='relu')

        self.objectness_logits = tf.keras.layers.Conv2D(filters=num_anchors, kernel_size=1, strides=1, padding="same")
        
        self.anchor_deltas = tf.keras.layers.Conv2D(filters=num_anchors * box_dim, kernel_size=1, strides=1, padding="same")

    def call(self, x, training=True):
        logits_and_deltas = []
        for feat_map in x:
            feat_map = self.rpn_conv(feat_map, training=training)
            logits_and_deltas.append(self.objectness_logits(feat_map, training=training))
            logits_and_deltas.append(self.anchor_deltas(feat_map, training=training))

        return logits_and_deltas

class TFROIHeads(tf.keras.Model):
    def __init__(self, output_size=(7,7)):
        super(TFROIHeads, self).__init__()

        self.ave_pool = tf.keras.layers.AveragePooling2D(pool_size=3, strides=2, padding="same")

        self.output_size = output_size
        self.pooler = tf.keras.layers.Average()

    def call(self, x, training=True):
        x = x[1:]
        crop_list = []
        for feat_map in x:
            x_pool = self.ave_pool(feat_map)
            crop_x = int(x_pool.shape[1] - self.output_size[0])
            crop_y = int(x_pool.shape[2] - self.output_size[1])
            x_cropped = tf.keras.layers.Cropping2D(cropping=((0, crop_x), (0, crop_y)))(x_pool)
            crop_list.append(x_cropped)
        return self.pooler(crop_list)


class RCNNHead(tf.keras.Model):
    def __init__(self, classes=80, box_dim=4):
        super(RCNNHead, self).__init__()

        self.box_head = tf.keras.Sequential()
        self.box_head.add(tf.keras.layers.Flatten())
        self.box_head.add(tf.keras.layers.Dense(1024, activation='relu'))
        self.box_head.add(tf.keras.layers.Dense(1024, activation='relu'))

        self.cls_score = tf.keras.layers.Dense(classes + 1)
        self.bbox_pred = tf.keras.layers.Dense(classes * box_dim)

    def call(self, x, training=True):
        x_box_feats = self.box_head(x)
        return self.cls_score(x_box_feats), self.bbox_pred(x_box_feats)