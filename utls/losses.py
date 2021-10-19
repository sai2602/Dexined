import tensorflow as tf


def cross_entropy_balanced(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32)

    count_neg = tf.reduce_sum(1. - y_true)
    count_pos = tf.reduce_sum(y_true)

    beta = count_neg/(count_pos + count_neg)

    pos_weight = beta/(1 - beta)

    cost = tf.nn.weighted_cross_entropy_with_logits(logits=y_pred, labels=y_true, pos_weight=pos_weight)

    cost = tf.reduce_mean(cost * (1 - beta))

    return tf.where(tf.equal(count_pos, 0.0), 0.0, cost)


def pixel_error(y_true, y_pred):
    pred = tf.cast(tf.greater(y_pred, 0.5), tf.int32)
    error = tf.cast(tf.not_equal(pred, tf.cast(y_true, tf.int32)), tf.float32)

    return tf.reduce_mean(error)
