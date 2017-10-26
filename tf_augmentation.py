import tensorflow as tf
import numpy as np
train_samples_per_epoch = 50000  
test_samples_per_epoch = 10000 

def get_batch_samples(img_obj, min_samples_in_queue, batch_size, shuffle_flag):
    '''''
    tf.train.shuffle_batch()函数用于随机地shuffling 队列中的tensors来创建batches(也即每次可以读取多个data文件中的样例构成一个batch)。这个函数向当前Graph中添加了下列对象：
    *创建了一个shuffling queue，用于把‘tensors’中的tensors压入该队列；
    *一个dequeue_many操作，用于根据队列中的数据创建一个batch；
    *创建了一个QueueRunner对象，用于启动一个进程压数据到队列
    capacity参数用于控制shuffling queue的最大长度；min_after_dequeue参数表示进行一次dequeue操作后队列中元素的最小数量，可以用于确保batch中
    元素的随机性；num_threads参数用于指定多少个threads负责压tensors到队列；enqueue_many参数用于表征是否tensors中的每一个tensor都代表一个样例
    tf.train.batch()与之类似，只不过顺序地出队列（也即每次只能从一个data文件中读取batch），少了随机性。
    '''


    if shuffle_flag == False:
        image_batch, label_batch = tf.train.batch(tensors=img_obj,
                                                  batch_size=batch_size,
                                                  num_threads=4,
                                                  capacity=min_samples_in_queue + 3 * batch_size)
    else:
        image_batch, label_batch = tf.train.shuffle_batch(tensors=img_obj,
                                                          batch_size=batch_size,
                                                          num_threads=4,
                                                          min_after_dequeue=min_samples_in_queue,
                                                          capacity=min_samples_in_queue + 3 * batch_size)
    return image_batch, tf.reshape(label_batch, shape=[batch_size])


def preprocess_input_data(img):
    '''''这部分程序用于对训练数据集进行‘数据增强’操作，通过增加训练集的大小来防止过拟合'''
    image = img  # 从文件名队列中读取一个tensor类型的图像
    tf.image_summary('raw_input_image', tf.reshape(image, [1, 32, 32, 3]))  # 输出预处理前图像的summary缓存对象
    #new_img = tf.random_crop(new_img, size=(fixed_height, fixed_width, 3))  # 从原图像中切割出子图像
    new_img = tf.image.random_brightness(image, max_delta=63)  # 随机调节图像的亮度
    new_img = tf.image.random_flip_left_right(new_img)  # 随机地左右翻转图像
    new_img = tf.image.random_contrast(new_img, lower=0.2, upper=1.8)  # 随机地调整图像对比度
    final_img = tf.image.per_image_whitening(new_img)  # 对图像进行whiten操作，目的是降低输入图像的冗余性，尽量去除输入特征间的相关性

    min_samples_ratio_in_queue = 0.4  # 用于确保读取到的batch中样例的随机性，使其覆盖到更多的类别、更多的数据文件！！！
    min_samples_in_queue = int(min_samples_ratio_in_queue * train_samples_per_epoch)
    return get_batch_samples([final_img, image.label], min_samples_in_queue, batch_size, shuffle_flag=True)
