def batch_norm(inputs, is_training,is_conv_out=True,decay = 0.999):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)

    if is_training:
        if is_conv_out:
            batch_mean, batch_var = tf.nn.moments(inputs,[0,1,2])
        else:
            batch_mean, batch_var = tf.nn.moments(inputs,[0])

        train_mean = tf.assign(pop_mean,
                               pop_mean * decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var,
                              pop_var * decay + batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs,
                batch_mean, batch_var, beta, scale, 0.001)
    else:
        return tf.nn.batch_normalization(inputs,
            pop_mean, pop_var, beta, scale, 0.001)
import tensorflow as tf


def residual_block(x,n_out, is_training, scope='res_block'):
    with tf.variable_scope(scope):
        if x.get_shape().as_list()[-1]==n_out:
            shortcut=tf.identity(x,name='shortcut')
        else:
            shortcut=tf.layers.conv2d(x, n_out, [1,1], padding='VALID', use_bias=False)
        y=tf.layers.conv2d(x,n_out,[3,3],padding='SAME',use_bias=False)
        y = tf.layers.batch_normalization(y,training=is_training)
        y = tf.nn.relu(y,name='relu_1')
        y = tf.layers.conv2d(y, n_out, [3, 3], padding='SAME', use_bias=False)
        y = tf.layers.batch_normalization(y,training=is_training)
        y = y + shortcut
        y = tf.nn.relu(y, name='relu_2')
    return y

image=tf.placeholder(tf.float32, shape=(None, 48,48,1))
label=tf.placeholder(tf.float32, [None, 5989])

class Res_model(object):
    def __init__(self,is_training=True):
        self.image=tf.placeholder(tf.float32, shape=(None, 48,48,1))
        self.label=tf.placeholder(tf.float32, [None, 5989])
        net,self.loss=self.build_network(is_training)
        self.predict=tf.nn.softmax(net)
        self.acc=self.cal_acc(net,self.label)

    def build_network(self,is_training=True):

        with tf.variable_scope('conv_1'):
            net=tf.layers.conv2d(self.image,64,[3,3],padding='SAME',use_bias=False)
        with tf.variable_scope('block_1'):
            net=  residual_block(net, 128, is_training, 'res_block_1')
            net = residual_block(net, 128, is_training, 'res_block_2')
            net = residual_block(net, 128, is_training, 'res_block_3')
        with tf.variable_scope('max_pooling_1'):
            net=tf.layers.max_pooling2d(net,[2,2],[2,2])
        with tf.variable_scope('block_2'):
            net = residual_block(net, 256, is_training, 'res_block_1')
            net = residual_block(net, 256, is_training, 'res_block_2')
            net = residual_block(net, 256, is_training, 'res_block_3')
        with tf.variable_scope('max_pooling_2'):
            net=tf.layers.max_pooling2d(net,[2,2],[2,2])
        with tf.variable_scope('block_3'):
            net = residual_block(net, 512, is_training, 'res_block_1')
            net = residual_block(net, 512, is_training, 'res_block_2')
            net = residual_block(net, 512, is_training, 'res_block_3')
        with tf.variable_scope('avg_pooling_1'):
            net=tf.layers.average_pooling2d(net,[5,5],[2,2])
        with tf.variable_scope('fully_connect'):
            net=tf.contrib.layers.flatten(net)
            net=tf.layers.dense(net,5989)

        return net,self.sce_loss(net,self.label)

    def sce_loss(self,logits,labels):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

    def cal_acc(self,logits, labels):
        prediction = tf.argmax(tf.nn.softmax(logits), 1)
        correct_prediction = tf.equal(prediction, tf.argmax(labels, 1))
        acc = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return acc
