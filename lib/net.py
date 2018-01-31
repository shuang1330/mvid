import tensorflow as tf
import tensorflow.contrib.layers as lays
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim import losses
from tensorflow.contrib.slim import arg_scope

def autoencoder(x):
    with arg_scope([slim.fully_connected],
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    weights_regularizer=slim.l2_regularizer(0.5)):
        net1 = slim.fully_connected(x,50)
        # net1 = slim.fully_connected(net1,50)
        # net2 = slim.fully_connected(net1,100)
        net = slim.fully_connected(net1,214)
    return net, net1


def feedforward_net(x):
    with arg_scope([slim.fully_connected],
    weights_initializer=tf.truncated_normal_initializer(stddev=0.01),
    weights_regularizer=slim.l2_regularizer(0.005)):
        net = slim.fully_connected(x,100)
        net = slim.fully_connected(net,10)
        # net = slim.fully_connected(net,10)
        net = slim.fully_connected(net,2)
        cls_prob = tf.nn.softmax(net, name="cls_prob")
    return net, cls_prob
