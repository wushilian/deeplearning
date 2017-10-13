import tensorflow as tf
slim=tf.contrib.slim
class Siamese:

    # Create model
    def __init__(self):
        width=50
        height=50
        self.num_class=34
        self.source = tf.placeholder(tf.float32, [None, width,height,1])
        self.target = tf.placeholder(tf.float32, [None, width, height, 1])
        self.y = tf.placeholder(tf.float32, [None,self.num_class])
        self.Is_same = tf.placeholder(tf.float32, [None])
        with tf.variable_scope("siamese") as scope:
            self.feature1,self.o1 = self.network(self.source)
            scope.reuse_variables()
            self.feature2,_ = self.network(self.target)#only need domain's label and predict
        self.loss = self.loss_with_step()
        self.acc=self.acc_cal()
    def network(self, x):
        conv1=slim.conv2d(x, 6,[5,5], scope='conv1')
        pool1=slim.max_pool2d(conv1, [2, 2], scope='pool1')
        conv2=slim.conv2d(pool1, 16, [5,5], scope='conv2')
        pool2=slim.max_pool2d(conv2, [2, 2], scope='pool2')
        conv3=slim.conv2d(pool2, 120, [5,5], scope='conv3')
        conv4 = slim.conv2d(conv3, 240, [3,3], scope='conv4')
        flat=slim.flatten(conv4)
        fc1=slim.fully_connected(flat, 2048, scope='fc1')
        drop1=slim.dropout(fc1,0.5,scope='dropout1')
        fc2=slim.fully_connected(drop1, 1024, scope='fc2')
        fc3 = slim.fully_connected(fc2, 512, scope='fc3')
        predict = slim.fully_connected(fc3, self.num_class, activation_fn=tf.nn.softmax, scope='pr0')

        #safe_exp = tf.clip_by_value(fc4, 1e-10, 10)

        return fc2,predict

    def loss_with_spring(self):
        margin = 5.0
        labels_t = self.Is_same
        labels_f = tf.subtract(1.0, self.y_, name="1-yi")  # labels_ = !labels;
        eucd2 = tf.pow(tf.subtract(self.feature1, self.feature2), 2)
        eucd2 = tf.reduce_sum(eucd2, 1)
        eucd = tf.sqrt(eucd2 + 1e-6, name="eucd")
        C = tf.constant(margin, name="C")
        # yi*||CNN(p1i)-CNN(p2i)||^2 + (1-yi)*max(0, C-||CNN(p1i)-CNN(p2i)||^2)
        pos = tf.multiply(labels_t, eucd2, name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.subtract(0.0,eucd2), name="yi_x_eucd2")
        # neg = tf.multiply(labels_f, tf.maximum(0.0, tf.subtract(C,eucd2)), name="Nyi_x_C-eucd_xx_2")
        neg = tf.multiply(labels_f, tf.pow(tf.maximum(tf.subtract(C, eucd), 0), 2), name="Nyi_x_C-eucd_xx_2")
        losses = tf.add(pos, neg, name="losses")
        loss = tf.reduce_mean(losses, name="loss")
        return loss
    def loss_with_step(self):#may be exist some error

        #safe_log=tf.clip_by_value(self.predict,1e-10,1e100)
        cross_entropy = -tf.reduce_sum(self.y*tf.log(self.predict))
        contrast_loss=self.loss_with_spring()
        return cross_entropy+contrast_loss
    def acc_cal(self):
        correct_prediction = tf.equal(tf.argmax(self.predict, 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return accuracy
