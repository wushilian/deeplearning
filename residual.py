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
def residual_block(x,n_out, is_training, scope='res_block'):
    with tf.variable_scope(scope):
        if x.get_shape().as_list()[-1]==n_out:
            shortcut=tf.identity(x,name='shortcut')
        else:
            shortcut=slim.conv2d(x, n_out, [1,1], scope='shortcut')
        y = slim.conv2d(x, n_out, [3, 3],activation_fn=None, scope='conv_1')
        y = batch_norm(y, n_out,is_training)
        y = tf.nn.relu(y,name='relu_1')
        y = slim.conv2d(x, n_out, [3, 3],activation_fn=None, scope='conv_2')
        y = batch_norm(y, n_out, is_training)
        y = y + shortcut
        y = tf.nn.relu(y, name='relu_2')
    return y
