# -*- coding:utf-8 -*-

import math
import cv2
import numpy as np

'''
    实现了dlib人脸的对齐策略
    但并不会像dlib那样使原始图像变形
    将人脸分为三种类型：中心，左，右
    根据人脸特征点对齐人脸
'''


class AlignCustom(object):
    def __init__(self):
        pass

    @staticmethod
    def get_pos(points):
        """获取position"""
        if abs(points[0] - points[2]) / abs(points[1] - points[2]) > 2:
            return 'Right'
        elif abs(points[1] - points[2]) / abs(points[0] - points[2]) > 2:
            return 'Left'
        return 'Center'

    @staticmethod
    def list2colmat(pts_list):
        """
        将列表转换为列矩阵
        :param pts_list: 输入的列表
        :return col_mat: 列矩阵
        """

        assert len(pts_list) > 0
        col_mat = []
        for i in range(len(pts_list)):
            col_mat.append(pts_list[i][0])
            col_mat.append(pts_list[i][1])
        col_mat = np.matrix(col_mat).transpose()
        return col_mat

    @staticmethod
    def transform_shapes(from_shape, to_shape):
        """转换shapes"""

        assert from_shape.shape[0] == to_shape.shape[0] and from_shape.shape[0] % 2 == 0

        sigma_from = 0.0
        cov = np.matrix([[0.0, 0.0], [0.0, 0.0]])

        # 计算mean和cov
        from_shape_points = from_shape.reshape(int(from_shape.shape[0] / 2), 2)
        to_shape_points = to_shape.reshape(int(to_shape.shape[0] / 2), 2)
        mean_from = from_shape_points.mean(axis=0)
        mean_to = to_shape_points.mean(axis=0)

        for i in range(from_shape_points.shape[0]):
            temp_dis = np.linalg.norm(from_shape_points[i] - mean_from)
            sigma_from += temp_dis * temp_dis
            cov += (to_shape_points[i].transpose() - mean_to.transpose()) * (from_shape_points[i] - mean_from)

        sigma_from = sigma_from / to_shape_points.shape[0]
        cov = cov / to_shape_points.shape[0]

        # 计算仿射矩阵
        s = np.matrix([[1.0, 0.0], [0.0, 1.0]])
        u, d, vt = np.linalg.svd(cov)

        if np.linalg.det(cov) < 0:
            if d[1] < d[0]:
                s[1, 1] = -1
            else:
                s[0, 0] = -1
        r = u * s * vt
        c = 1.0
        if sigma_from != 0:
            c = 1.0 / sigma_from * np.trace(np.diag(d) * s)

        tran_b = mean_to.transpose() - c * r * mean_from.transpose()
        tran_m = c * r

        return tran_m, tran_b

    def align(self,
              desired_size,
              img_face,
              landmarks,
              padding=0.1):
        """
        以BGR格式对齐人脸.
        :param desired_size: 所需的图像大小
        :param img_face: 检测的人脸图像
        :param landmarks: 人脸特征点
        :param padding: 卷积方式
        """

        shape = []
        for k in range(int(len(landmarks) / 2)):
            shape.append(landmarks[k])
            shape.append(landmarks[k + 5])

        if padding > 0:
            padding = padding
        else:
            padding = 0

        # 面部点的平均位置
        mean_face_shape_x = [0.224152, 0.75610125, 0.490127, 0.254149, 0.726104]
        mean_face_shape_y = [0.2119465, 0.2119465, 0.628106, 0.780233, 0.780233]

        from_points = []
        to_points = []
        for i in range(int(len(shape) / 2)):
            x = (padding + mean_face_shape_x[i]) / (2 * padding + 1) * desired_size
            y = (padding + mean_face_shape_y[i]) / (2 * padding + 1) * desired_size
            to_points.append([x, y])
            from_points.append([shape[2 * i], shape[2 * i + 1]])

        # 将点转换为Mat
        from_mat = self.list2colmat(from_points)
        to_mat = self.list2colmat(to_points)

        # 计算相似转换
        tran_m, tran_b = self.transform_shapes(from_mat, to_mat)
        probe_vec = np.matrix([1.0, 0.0]).transpose()
        probe_vec = tran_m * probe_vec
        scale = np.linalg.norm(probe_vec)
        angle = 180.0 / math.pi * math.atan2(probe_vec[1, 0], probe_vec[0, 0])

        from_center = [(shape[0] + shape[2]) / 2.0, (shape[1] + shape[3]) / 2.0]
        to_center = [0, 0]
        to_center[1] = desired_size * 0.4
        to_center[0] = desired_size * 0.5
        ex = to_center[0] - from_center[0]
        ey = to_center[1] - from_center[1]

        rot_mat = cv2.getRotationMatrix2D((from_center[0], from_center[1]), -1 * angle, scale)
        rot_mat[0][2] += ex
        rot_mat[1][2] += ey

        result = cv2.warpAffine(img_face, rot_mat, (desired_size, desired_size))
        return result, self.get_pos(landmarks)
