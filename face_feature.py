# -*- coding:utf-8 -*-

import os
import re
import tensorflow as tf
import inception_resnet_v1 as resnet
import numpy as np

"""
运行预训练模型以提取128D人脸特征
"""


class FaceFeature(object):
    def __init__(self,
                 face_rec_graph,
                 model_dir):

        """
        :param face_rec_graph: 输入人脸的graph
        :param model_dir: 用于识别人脸的model的路径
        """

        model_path = model_dir + get_model_name(model_dir)
        # print(model_path)
        print('正在加载model...')
        config = tf.ConfigProto()
        # config.allow_soft_placement = True  # 刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
        config.gpu_options.allow_growth = True
        with face_rec_graph.graph.as_default():
            self.sess = tf.Session(config=config)

            # 默认输入的NN尺寸为128 x 128 x 3
            self.x = tf.placeholder(dtype=tf.float32, shape=[None, 128, 128, 3])

            self.embeddings = tf.nn.l2_normalize(
                                        resnet.inference(self.x, 0.6, phase_train=False)[0], 1, 1e-10)

            saver = tf.train.Saver()  # 创建一个Saver来管理模型中的所有变量
            saver.restore(self.sess, model_path)
            print('model加载完毕！')

    def get_features(self, input_imgs):  # 图像预处理
        images = load_data_list(input_imgs,128)
        return self.sess.run(self.embeddings, feed_dict={self.x: images})


def get_model_name(model_dir):  # 使用正则表达式匹配model的名称
    files = os.listdir(model_dir)
    model_name = ''
    pattern = r'(^model-[\w\- ]+.ckpt-(\d+))'
    for f in files:
        name_str = re.match(pattern, f)
        if name_str is not None:
            model_name = name_str.groups()[0]

    return model_name


def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0 / np.sqrt(x.size))
    y = np.multiply(np.subtract(x, mean), 1 / std_adj)
    return y


def load_data_list(img_list,
                   image_size,
                   do_prewhiten=True):
    images = np.zeros((len(img_list), image_size, image_size, 3))
    i = 0
    for img in img_list:
        if img is not None:
            if do_prewhiten:
                img = prewhiten(img)
            images[i, :, :, :] = img
            i += 1
    return images
