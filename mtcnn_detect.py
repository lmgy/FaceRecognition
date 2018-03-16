# -*- coding:utf-8 -*-

import os
import cv2
import numpy as np
import tensorflow as tf
from six import string_types, iteritems


'''
使用Tensorflow实现的MTCNN人脸检测算法
'''


class MTCNNDetect(object):
    def __init__(self, face_rec_graph, scale_factor, model_dir):
        """
        :param face_rec_graph: 人脸识别graph
        :param scale_factor:比例
        :param model_dir: 模型的目录
        """

        self.threshold = [0.6, 0.7, 0.7]  # 检测的阈值
        self.factor = 0.709  # 默认0.709
        self.scale_factor = scale_factor  # 比例

        config = tf.ConfigProto()
        # config.allow_soft_placement = True  # 刚一开始分配少量的GPU容量，然后按需慢慢的增加，由于不会释放内存，所以会导致碎片
        config.gpu_options.allow_growth = True  # 控制GPU资源使用率

        with face_rec_graph.graph.as_default():
            print('正在加载MTCNN人脸检测model...')
            self.sess = tf.Session(config=config)  # 根据需求分配显存，防止占用所有显存

            if not model_dir:
                model_dir, _ = os.path.split(os.path.realpath(__file__))

            with tf.variable_scope('pnet'):
                data = tf.placeholder(tf.float32, (None, None, None, 3), 'input')
                pnet = PNet({'data': data})
                pnet.load(os.path.join(model_dir, 'det1.npy'), self.sess)
            with tf.variable_scope('rnet'):
                data = tf.placeholder(tf.float32, (None, 24, 24, 3), 'input')
                rnet = RNet({'data': data})
                rnet.load(os.path.join(model_dir, 'det2.npy'), self.sess)
            with tf.variable_scope('onet'):
                data = tf.placeholder(tf.float32, (None, 48, 48, 3), 'input')
                onet = ONet({'data': data})
                onet.load(os.path.join(model_dir, 'det3.npy'), self.sess)

            self.pnet = lambda img: self.sess.run(('pnet/conv4-2/BiasAdd:0', 'pnet/prob1:0'),
                                                  feed_dict={'pnet/input:0': img})
            self.rnet = lambda img: self.sess.run(('rnet/conv5-2/conv5-2:0', 'rnet/prob1:0'),
                                                  feed_dict={'rnet/input:0': img})
            self.onet = lambda img: self.sess.run(('onet/conv6-2/conv6-2:0', 'onet/conv6-3/conv6-3:0', 'onet/prob1:0'),
                                                  feed_dict={'onet/input:0': img})
            print('MTCNN人脸检测model加载完毕！')

    def detect_face(self, img, minsize):
        """
        :param img: 输入的图像
        :param minsize: 最小检测的人脸大小
        """

        if self.scale_factor > 1:
            img = cv2.resize(img, (int(len(img[0])/self.scale_factor), int(len(img)/self.scale_factor)))

        factor_count = 0
        total_boxes = np.empty((0, 9))
        points = []
        h = img.shape[0]
        w = img.shape[1]
        minl = np.amin([h, w])
        m = 12.0 / minsize
        minl = minl * m

        scales = []
        while minl >= 12:
            scales += [m * np.power(self.factor, factor_count)]
            minl = minl * self.factor
            factor_count += 1

        # 第一阶段
        for j in range(len(scales)):
            scale = scales[j]
            hs = int(np.ceil(h * scale))
            ws = int(np.ceil(w * scale))
            im_data = imresample(img, (hs, ws))
            im_data = (im_data - 127.5) * 0.0078125
            img_x = np.expand_dims(im_data, 0)
            img_y = np.transpose(img_x, (0, 2, 1, 3))
            out = self.pnet(img_y)
            out0 = np.transpose(out[0], (0, 2, 1, 3))
            out1 = np.transpose(out[1], (0, 2, 1, 3))

            boxes, _ = generate_boundingbox(out1[0, :, :, 1].copy(), out0[0, :, :, :].copy(), scale, self.threshold[0])

            pick = nms(boxes.copy(), 0.5, 'Union')
            if boxes.size > 0 and pick.size > 0:
                boxes = boxes[pick, :]
                total_boxes = np.append(total_boxes, boxes, axis=0)

        numbox = total_boxes.shape[0]

        if numbox > 0:
            pick = nms(total_boxes.copy(), 0.7, 'Union')
            total_boxes = total_boxes[pick, :]
            regw = total_boxes[:, 2] - total_boxes[:, 0]
            regh = total_boxes[:, 3] - total_boxes[:, 1]
            qq1 = total_boxes[:, 0] + total_boxes[:, 5] * regw
            qq2 = total_boxes[:, 1] + total_boxes[:, 6] * regh
            qq3 = total_boxes[:, 2] + total_boxes[:, 7] * regw
            qq4 = total_boxes[:, 3] + total_boxes[:, 8] * regh
            total_boxes = np.transpose(np.vstack([qq1, qq2, qq3, qq4, total_boxes[:, 4]]))
            total_boxes = rerec(total_boxes.copy())
            total_boxes[:, 0:4] = np.fix(total_boxes[:, 0:4]).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)

        numbox = total_boxes.shape[0]

        if numbox > 0:

            # 第二阶段
            tmp_img = np.zeros((24, 24, 3, numbox))

            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tmp_img[:, :, :, k] = imresample(tmp, (24, 24))
                else:
                    return np.empty()

            tmp_img = (tmp_img - 127.5) * 0.0078125
            tmp_img1 = np.transpose(tmp_img, (3, 1, 0, 2))
            out = self.rnet(tmp_img1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            score = out1[1, :]
            ipass = np.where(score > self.threshold[1])
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
            mv = out0[:, ipass[0]]
            if total_boxes.shape[0] > 0:
                pick = nms(total_boxes, 0.7, 'Union')
                total_boxes = total_boxes[pick, :]
                total_boxes = bbreg(total_boxes.copy(), np.transpose(mv[:, pick]))
                total_boxes = rerec(total_boxes.copy())

        numbox = total_boxes.shape[0]

        if numbox > 0:
            # 第三阶段
            total_boxes = np.fix(total_boxes).astype(np.int32)
            dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph = pad(total_boxes.copy(), w, h)
            tmp_img = np.zeros((48, 48, 3, numbox))

            for k in range(0, numbox):
                tmp = np.zeros((int(tmph[k]), int(tmpw[k]), 3))
                tmp[dy[k] - 1:edy[k], dx[k] - 1:edx[k], :] = img[y[k] - 1:ey[k], x[k] - 1:ex[k], :]
                if tmp.shape[0] > 0 and tmp.shape[1] > 0 or tmp.shape[0] == 0 and tmp.shape[1] == 0:
                    tmp_img[:, :, :, k] = imresample(tmp, (48, 48))
                else:
                    return np.empty()

            tmp_img = (tmp_img - 127.5) * 0.0078125
            tmp_img1 = np.transpose(tmp_img, (3, 1, 0, 2))
            out = self.onet(tmp_img1)
            out0 = np.transpose(out[0])
            out1 = np.transpose(out[1])
            out2 = np.transpose(out[2])
            score = out2[1, :]
            points = out1
            ipass = np.where(score > self.threshold[2])
            points = points[:, ipass[0]]
            total_boxes = np.hstack([total_boxes[ipass[0], 0:4].copy(), np.expand_dims(score[ipass].copy(), 1)])
            mv = out0[:, ipass[0]]

            w = total_boxes[:, 2] - total_boxes[:, 0] + 1
            h = total_boxes[:, 3] - total_boxes[:, 1] + 1
            points[0:5, :] = np.tile(w, (5, 1)) * points[0:5, :] + np.tile(total_boxes[:, 0], (5, 1)) - 1
            points[5:10, :] = np.tile(h, (5, 1)) * points[5:10, :] + np.tile(total_boxes[:, 1], (5, 1)) - 1

            if total_boxes.shape[0] > 0:
                total_boxes = bbreg(total_boxes.copy(), np.transpose(mv))
                pick = nms(total_boxes.copy(), 0.7, 'Min')
                total_boxes = total_boxes[pick, :]
                points = points[:, pick]

        simple_points = np.transpose(points)  # points存储在一个数据结构中，更容易转置

        rects = [
            (max(0, (int(rect[0]))) * self.scale_factor,
                max(0, int(rect[1])) * self.scale_factor,
                int(rect[2] - rect[0]) * self.scale_factor,
                int(rect[3] - rect[1]) * self.scale_factor) for rect in total_boxes]

        return rects, simple_points * self.scale_factor


def layer(op):

    def layer_decorated(self, *args, **kwargs):  # 如果没有提供名字，则自动设置一个
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        if len(self.terminals) == 0:  # 找出layer的输入
            raise RuntimeError('No input variables found for layer %s.' % name)
        elif len(self.terminals) == 1:
            layer_input = self.terminals[0]
        else:
            layer_input = list(self.terminals)

        layer_output = op(self, layer_input, *args, **kwargs)  # 执行操作并获得输出

        self.layers[name] = layer_output  # 添加到layer LUT

        self.feed(layer_output)

        return self  # 向chained calls返回self

    return layer_decorated


class Network(object):

    def __init__(self, inputs, trainable=True):

        self.inputs = inputs
        self.terminals = []
        self.layers = dict(inputs)
        self.trainable = trainable

        self.setup()

    def setup(self):  # 构建网络
        raise NotImplementedError('Must be implemented by the subclass')

    def load(self, data_path, session, ignore_missing=False):
        """
        加载网络的权重
        :param data_path: numpy serialized weights的路径
        :param session: 当前的TensorFlow session
        :param ignore_missing: 如果为True，则忽略缺失图层的serialized weights
        """

        data_dict = np.load(data_path, encoding='latin1').item()

        for op_name in data_dict:
            with tf.variable_scope(op_name, reuse=True):
                for param_name, data in iteritems(data_dict[op_name]):
                    try:
                        var = tf.get_variable(param_name)
                        session.run(var.assign(data))
                    except ValueError:
                        if not ignore_missing:
                            raise

    def feed(self, *args):
        """
        通过替换终端节点来设置下一个操作的输入
        :param args: layer的名称或者是实际的layer
        """

        assert len(args) != 0
        self.terminals = []
        for fed_layer in args:
            if isinstance(fed_layer, string_types):
                try:
                    fed_layer = self.layers[fed_layer]
                except KeyError:
                    raise KeyError('Unknown layer name fed: %s' % fed_layer)
            self.terminals.append(fed_layer)
        return self

    def get_output(self):
        """返回当前的网络输出"""

        return self.terminals[-1]

    def get_unique_name(self, prefix):
        """
        基于type-prefix自动生成layer名称。
        :return 输入的prefix的index-suffixed唯一名称
        """

        ident = sum(t.startswith(prefix) for t, _ in self.layers.items()) + 1
        return '%s_%d' % (prefix, ident)

    def make_var(self, name, shape):
        """创建一个新的TensorFlow变量"""

        return tf.get_variable(name, shape, trainable=self.trainable)

    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    def conv(self,
             inp,
             k_h,
             k_w,
             c_o,
             s_h,
             s_w,
             name,
             relu=True,
             padding='SAME',
             group=1,
             biased=True):

        self.validate_padding(padding)  # 确认padding是可接受的

        c_i = int(inp.get_shape()[-1])  # 获取输入中的channel数量

        # 验证参数是否有效
        assert c_i % group == 0
        assert c_o % group == 0

        def convolve(i, k):
            """input和kernel的卷积"""

            return tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding, use_cudnn_on_gpu=True)

        with tf.variable_scope(name) as scope:
            kernel = self.make_var('weights', shape=[k_h, k_w, c_i // group, c_o])
            output = convolve(inp, kernel)  # 卷积input和kernel

            if biased:  # 添加biases
                biases = self.make_var('biases', [c_o])
                output = tf.nn.bias_add(output, biases)
            if relu:
                output = tf.nn.relu(output, name=scope.name)  # ReLU非线性

            return output

    @layer
    def prelu(self, inp, name):
        with tf.variable_scope(name):
            i = int(inp.get_shape()[-1])
            alpha = self.make_var('alpha', shape=(i,))
            output = tf.nn.relu(inp) + tf.multiply(alpha, -tf.nn.relu(-inp))
        return output

    @layer
    def max_pool(self, inp, k_h, k_w, s_h, s_w, name, padding='SAME'):
        self.validate_padding(padding)
        return tf.nn.max_pool(inp,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def fc(self, inp, num_out, name, relu=True):
        with tf.variable_scope(name):
            input_shape = inp.get_shape()

            # input为spatial，因此需要首先vectorize
            if input_shape.ndims == 4:
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= int(d)
                feed_in = tf.reshape(inp, [-1, dim])
            else:
                feed_in, dim = (inp, input_shape[-1].value)

            weights = self.make_var('weights', shape=[dim, num_out])
            biases = self.make_var('biases', [num_out])
            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name=name)
            return fc

    @layer
    def softmax(self, target, axis, name=None):
        max_axis = tf.reduce_max(target, axis, keep_dims=True)
        target_exp = tf.exp(target - max_axis)
        normalize = tf.reduce_sum(target_exp, axis, keep_dims=True)
        softmax = tf.div(target_exp, normalize, name)
        return softmax


class PNet(Network):
    def setup(self):
        (self.feed('data')
            .conv(3, 3, 10, 1, 1, padding='VALID', relu=False, name='conv1')
            .prelu(name='PReLU1')
            .max_pool(2, 2, 2, 2, name='pool1')
            .conv(3, 3, 16, 1, 1, padding='VALID', relu=False, name='conv2')
            .prelu(name='PReLU2')
            .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv3')
            .prelu(name='PReLU3')
            .conv(1, 1, 2, 1, 1, relu=False, name='conv4-1')
            .softmax(3, name='prob1'))

        (self.feed('PReLU3')
            .conv(1, 1, 4, 1, 1, relu=False, name='conv4-2'))


class RNet(Network):
    def setup(self):
        (self.feed('data')
            .conv(3, 3, 28, 1, 1, padding='VALID', relu=False, name='conv1')
            .prelu(name='prelu1')
            .max_pool(3, 3, 2, 2, name='pool1')
            .conv(3, 3, 48, 1, 1, padding='VALID', relu=False, name='conv2')
            .prelu(name='prelu2')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            .conv(2, 2, 64, 1, 1, padding='VALID', relu=False, name='conv3')
            .prelu(name='prelu3')
            .fc(128, relu=False, name='conv4')
            .prelu(name='prelu4')
            .fc(2, relu=False, name='conv5-1')
            .softmax(1, name='prob1'))

        (self.feed('prelu4')
            .fc(4, relu=False, name='conv5-2'))


class ONet(Network):
    def setup(self):
        (self.feed('data')
            .conv(3, 3, 32, 1, 1, padding='VALID', relu=False, name='conv1')
            .prelu(name='prelu1')
            .max_pool(3, 3, 2, 2, name='pool1')
            .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv2')
            .prelu(name='prelu2')
            .max_pool(3, 3, 2, 2, padding='VALID', name='pool2')
            .conv(3, 3, 64, 1, 1, padding='VALID', relu=False, name='conv3')
            .prelu(name='prelu3')
            .max_pool(2, 2, 2, 2, name='pool3')
            .conv(2, 2, 128, 1, 1, padding='VALID', relu=False, name='conv4')
            .prelu(name='prelu4')
            .fc(256, relu=False, name='conv5')
            .prelu(name='prelu5')
            .fc(2, relu=False, name='conv6-1')
            .softmax(1, name='prob1'))

        (self.feed('prelu5')
            .fc(4, relu=False, name='conv6-2'))

        (self.feed('prelu5')
            .fc(10, relu=False, name='conv6-3'))


def bbreg(boundingbox, reg):
    """校准bounding boxes"""

    if reg.shape[1] == 1:
        reg = np.reshape(reg, (reg.shape[2], reg.shape[3]))

    w = boundingbox[:, 2] - boundingbox[:, 0] + 1
    h = boundingbox[:, 3] - boundingbox[:, 1] + 1
    b1 = boundingbox[:, 0] + reg[:, 0] * w
    b2 = boundingbox[:, 1] + reg[:, 1] * h
    b3 = boundingbox[:, 2] + reg[:, 2] * w
    b4 = boundingbox[:, 3] + reg[:, 3] * h
    boundingbox[:, 0:4] = np.transpose(np.vstack([b1, b2, b3, b4]))
    return boundingbox


def generate_boundingbox(imap, reg, scale, t):
    """使用heatmap生成bounding boxes"""

    stride = 2
    cellsize = 12

    imap = np.transpose(imap)
    dx1 = np.transpose(reg[:, :, 0])
    dy1 = np.transpose(reg[:, :, 1])
    dx2 = np.transpose(reg[:, :, 2])
    dy2 = np.transpose(reg[:, :, 3])
    y, x = np.where(imap >= t)
    if y.shape[0] == 1:
        dx1 = np.flipud(dx1)
        dy1 = np.flipud(dy1)
        dx2 = np.flipud(dx2)
        dy2 = np.flipud(dy2)
    score = imap[(y, x)]
    reg = np.transpose(np.vstack([dx1[(y, x)], dy1[(y, x)], dx2[(y, x)], dy2[(y, x)]]))

    if reg.size == 0:
        reg = np.empty((0, 3))

    bb = np.transpose(np.vstack([y, x]))
    q1 = np.fix((stride * bb + 1) / scale)
    q2 = np.fix((stride * bb + cellsize - 1 + 1) / scale)
    boundingbox = np.hstack([q1, q2, np.expand_dims(score, 1), reg])
    return boundingbox, reg


def nms(boxes, threshold, method):

    if boxes.size == 0:
        return np.empty((0, 3))

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    s = boxes[:, 4]
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    I = np.argsort(s)
    pick = np.zeros_like(s, dtype=np.int16)
    counter = 0

    while I.size > 0:
        i = I[-1]
        pick[counter] = i
        counter += 1
        idx = I[0:-1]
        xx1 = np.maximum(x1[i], x1[idx])
        yy1 = np.maximum(y1[i], y1[idx])
        xx2 = np.minimum(x2[i], x2[idx])
        yy2 = np.minimum(y2[i], y2[idx])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if method is 'Min':
            o = inter / np.minimum(area[i], area[idx])
        else:
            o = inter / (area[i] + area[idx] - inter)

        I = I[np.where(o <= threshold)]
    pick = pick[0:counter]
    return pick


def pad(total_boxes, w, h):
    """
    计算填充坐标（将boxes填充为square）
    :param total_boxes: boxes
    :param w: 宽度
    :param h: 高度
    """

    tmpw = (total_boxes[:, 2] - total_boxes[:, 0] + 1).astype(np.int32)
    tmph = (total_boxes[:, 3] - total_boxes[:, 1] + 1).astype(np.int32)
    numbox = total_boxes.shape[0]

    dx = np.ones((numbox), dtype=np.int32)
    dy = np.ones((numbox), dtype=np.int32)
    edx = tmpw.copy().astype(np.int32)
    edy = tmph.copy().astype(np.int32)

    x = total_boxes[:, 0].copy().astype(np.int32)
    y = total_boxes[:, 1].copy().astype(np.int32)
    ex = total_boxes[:, 2].copy().astype(np.int32)
    ey = total_boxes[:, 3].copy().astype(np.int32)

    tmp = np.where(ex > w)
    edx.flat[tmp] = np.expand_dims(-ex[tmp] + w + tmpw[tmp], 1)
    ex[tmp] = w

    tmp = np.where(ey > h)
    edy.flat[tmp] = np.expand_dims(-ey[tmp] + h + tmph[tmp], 1)
    ey[tmp] = h

    tmp = np.where(x < 1)
    dx.flat[tmp] = np.expand_dims(2 - x[tmp], 1)
    x[tmp] = 1

    tmp = np.where(y < 1)
    dy.flat[tmp] = np.expand_dims(2 - y[tmp], 1)
    y[tmp] = 1

    return dy, edy, dx, edx, y, ey, x, ex, tmpw, tmph


def rerec(bboxA):
    """
    将bboxA转换为square
    :param bboxA: bboxA
    :return: square
    """

    h = bboxA[:, 3] - bboxA[:, 1]
    w = bboxA[:, 2] - bboxA[:, 0]
    l = np.maximum(w, h)
    bboxA[:, 0] = bboxA[:, 0] + w * 0.5 - l * 0.5
    bboxA[:, 1] = bboxA[:, 1] + h * 0.5 - l * 0.5
    bboxA[:, 2:4] = bboxA[:, 0:2] + np.transpose(np.tile(l, (2, 1)))
    return bboxA


def imresample(img, sz):
    """
    图像尺寸变换
    :param img: 图像数据
    :param sz: 想要的图像尺寸
    :return: 转换后的图像数据
    """

    im_data = cv2.resize(img, (sz[1], sz[0]), interpolation=cv2.INTER_AREA)
    return im_data
