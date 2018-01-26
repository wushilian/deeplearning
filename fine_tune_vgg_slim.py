#https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py vgg源码
#http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz vgg模型下载地址
#此种方法适用图片较大，或希望在vgg基础上fintune，不修改网络结构
import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg
slim=tf.contrib.slim
sess=tf.Session()
img=tf.placeholder(tf.float32,[None,224,224,3])
vgg16=vgg.vgg_16(inputs=img,num_classes=1000,is_training=True)
variables_to_restore = slim.get_model_variables()
variables_to_restore = slim.get_variables_to_restore(exclude=['fc6', 'fc7', 'fc8'])#得到此时图内所有变量名,并去除全连接
restorer = tf.train.Saver(variables_to_restore)#恢复模型
restorer.restore(sess,'vgg_16.ckpt')


#这种方法修改vgg的网络结构，更加灵活，可以在vgg基础上构建自己的网络模型
import tensorflow as tf
import numpy as np

slim=tf.contrib.slim
def vgg_16(inputs,scope='vgg_16',):

#修改模型结构，从而只载入vgg前几层
  with tf.variable_scope(scope, 'vgg_16', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.fully_connected, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      net = slim.repeat(inputs, 2, slim.conv2d, 64, [3, 3], scope='conv1')
      net = slim.max_pool2d(net, [2, 2], scope='pool1')
      net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
      net = slim.max_pool2d(net, [2, 2], scope='pool2')
      net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
      net = slim.max_pool2d(net, [2, 2], scope='pool3')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
      net = slim.max_pool2d(net, [2, 2], scope='pool4')
      net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
      return net
sess=tf.Session()
img=tf.placeholder(tf.float32,[None,28,28,3])
vgg=vgg_16(img)
variables_to_restore = slim.get_variables_to_restore()#得到此时图内所有变量名
restorer = tf.train.Saver(variables_to_restore)#恢复模型
restorer.restore(sess,'vgg_16.ckpt')
