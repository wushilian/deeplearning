import sys
import xml.etree.cElementTree as ET
import numpy as np
import tensorflow as tf
import glob
import cv2
import scipy.io as sio
import os
import re
from matplotlib import pyplot as plt
from tqdm import *

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))





def convert_synthtext_tfrecords():
    zz='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    data_dir = '/data/SynthText/SynthText'
    mat = sio.loadmat(os.path.join(data_dir, 'gt.mat'))

    writer = tf.python_io.TFRecordWriter(path='/data/SynthText/synthtext_train.tfrecords')
    imnames = mat['imnames'][0]
    charBB = mat['charBB'][0]  # (2,4,n_chars)
    wordBB = mat['wordBB'][0]  # (2,4,n_words)
    txt = mat['txt'][0]
    #len(imnames)
    for i in tqdm(range(len(imnames))):
        box_list=[]
        img_name=imnames[i][0]
        img= cv2.imread(os.path.join(data_dir, img_name))
        img_height,img_width=img.shape[0],img.shape[1]

        temp = txt[i]
        for j in range(len(temp)):
            temp[j] = temp[j].replace(' ', '').replace('\n', '')
        temp = ''.join(temp)
        bbox = np.transpose(charBB[i], axes=(2, 1, 0)).astype(np.int32)
        if not len(bbox)==len(temp):
            print('label length is not eaual to bbox')
            continue
        for j in range(len(bbox)):

            if temp[j] in zz:
                tmp_box=np.reshape(bbox[j],(-1))
                tmp_box=list(tmp_box)
                index=zz.index(temp[j])+1

                tmp_box.append(index)

                box_list.append(tmp_box)
        gtbox_label = np.array(box_list, dtype=np.int32)

        feature = tf.train.Features(feature={
            'img_name': _bytes_feature(img_name.encode()),
            'img_height': _int64_feature(img_height),
            'img_width': _int64_feature(img_width),
            'img': _bytes_feature(img.tostring()),
            'gtboxes_and_label': _bytes_feature(gtbox_label.tostring()),
            'num_objects': _int64_feature(gtbox_label.shape[0])
        })

        example = tf.train.Example(features=feature)

        writer.write(example.SerializeToString())

    writer.close()

def viz():
    data_dir = '/data/SynthText/SynthText'
    mat = sio.loadmat(os.path.join(data_dir, 'gt.mat'))

    imnames = mat['imnames'][0]
    charBB = mat['charBB'][0]  # (2,4,n_chars)
    wordBB = mat['wordBB'][0]  # (2,4,n_words)
    txt = mat['txt'][0]

    im = cv2.imread(os.path.join(data_dir, imnames[0][0]))

    bbox = np.transpose(charBB[0], axes=(2, 1, 0)).astype(np.int32)
    temp = txt[0]
    for i in range(len(temp)):
        temp[i] = temp[i].replace(' ', '').replace('\n', '')
    temp = ''.join(temp)
    for i in range(len(bbox)):
        im = cv2.putText(im, temp[i], (bbox[i, 0, 0], bbox[i, 0, 1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 1,
                         cv2.LINE_4)
    im=cv2.polylines(im,bbox,isClosed=True,color=(0,255,255),thickness=3)
    plt.imshow(im)
    plt.show()

    print(temp, len(temp))
if __name__ == '__main__':
    convert_synthtext_tfrecords()
   #viz()




