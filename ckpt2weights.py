import tensorflow as tf
structure=['Conv2D','Activation','MaxPooling2D','Conv2D','Activation','MaxPooling2D','Conv2D',
           'Activation','Flatten','Dense','Activation','Drop','Dense','Activation','Dense','Activation']
parameter=['5 5 1 6','relu','','5 5 6 16','relu','','5 5 16 120','relu',
           '','7680 512','relu','','512 256','relu','256 34','softmax']

def get_weights():
    sess = tf.InteractiveSession()

    cnn = model.cnn_ocr()

    # train_step = tf.train.AdadeltaOptimizer(0.01).minimize(cnn.loss)
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    path = cfg.pretrained_path
    saver.restore(sess, path)
    print('model has been restored')
    variable = sess.run(tf.global_variables())
    #print(variable[0].shape)
    return variable
def get_txt():
    variable=get_weights()
    fp=open('lenet5.weights','w')
    fp.writelines('layers 16\n')
    #num_layer=0
    z=0
    for i in range(len(structure)):
        print(i)
        if structure[i]=='Conv2D' or structure[i]=='Dense':
            fp.writelines('layer '+str(i)+' '+structure[i]+'\n')
            fp.writelines(parameter[i]+'\n')
            fp.writelines('weight:\n')
            t = np.reshape(variable[z], [-1])
            text = str(t.tolist()).replace(',', ' ').replace('[', '').replace(']', '')
            fp.writelines(text+'\n')
            z+=1
            fp.writelines('bias:\n')
            t = np.reshape(variable[z], [-1])
            text = str(t.tolist()).replace(',', ' ').replace('[', '').replace(']', '')
            fp.writelines(text + '\n')
            z+=1
        elif structure[i]=='Activation':
            fp.writelines('layer '+str(i)+' Activation\n')
            fp.writelines(parameter[i]+'\n')
        else:
            fp.writelines('layer '+str(i)+' '+structure[i]+'\n')
    fp.close()

get_txt()
