import tensorflow as tf
import tensorflow.contrib.layers as lays
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

def autoencoder(x):
    with arg_scope([slim.fully_connected],
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    weights_regularizer=slim.l2_regularizer(0.0005)):
        net = slim.fully_connected(x,2)
        net = slim.fully_connected(net,180)
    return net


def feedforward_net(x):
    with arg_scope([slim.fully_connected],
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    weights_regularizer=slim.l2_regularizer(0.005)):
        net = slim.fully_connected(x,90)
        net = slim.fully_connected(net,10)
        net = slim.fully_connected(net,2)
    return net
