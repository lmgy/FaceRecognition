# -*- coding:utf-8 -*-

import os
import sys
import time
import json
import cv2
import numpy as np
import tensorflow as tf
import gc
from align_custom import AlignCustom
from face_feature import FaceFeature
from mtcnn_detect import MTCNNDetect
from PIL import Image, ImageDraw, ImageFont

"""
识别原理：
    - >捕捉摄像头的图像
    - >检测人脸的区域
    - >裁剪并对齐
  - >将每个裁剪的人脸为三种类型：中心，左，右
    - >提取128D vectors （人脸特征）
  - >根据脸部位置的类型搜索数据集中的匹配主体。
  - >与128D vectors 屏幕上的最短距离的预先生成的128D vectors 最有可能匹配 （默认距离阈值0.6，相似度百分比阈值70％）
"""


def camera_recognize(database_path, min_rec_size):

    """
    通过摄像头识别出图像中的人物姓名或ID
    :param database_path: FaceRecognition_128D.json文件路径
    :param min_rec_size: 能被摄像头识别到的最小人脸大小
    """

    print('正在启动摄像头...')
    camera = cv2.VideoCapture(0)  # 启动摄像头
    print('按q键退出')
    while True:
        time_start = time.time()

        _, frame = camera.read()
        rects, landmarks = face_detect.detect_face(frame, min_rec_size)  # 检测人脸时，默认能被检测到的最小人脸大小为32 x 32
        aligns = []
        positions = []

        for (i, rect) in enumerate(rects):
            aligned_face, face_pos = aligner.align(128, frame, landmarks[i])
            aligns.append(aligned_face)
            positions.append(face_pos)
        features_arr = extract_feature.get_features(aligns)
        recog_data = identify_people(database_path, features_arr, positions, thres=THRES, percent_thres=PERCENT_THRES)
        for (i, rect) in enumerate(rects):
            cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
            pil_im = Image.fromarray(cv2_im)
            draw = ImageDraw.Draw(pil_im)
            font = ImageFont.truetype('simsun.ttc', 25)  # simsun.ttc为宋体，如果系统没有该字体，请自行添加,字体大小25
            draw.text((rect[0], rect[1]), recog_data[i][0] + ' - ' + str(recog_data[i][1])[0:4] + '%',
                      (255, 255, 255), font=font)  # 支持中文字符，取结果字符串的前3位
            frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

            if recog_data[i][1] < 70:
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color)  # 框选出检测出的人脸

        time_end = time.time()
        time_seconds = time_end - time_start
        fps = str(1 / time_seconds)[0:4]  # 计算当前的fps

        # 显示fps
        frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cv2.putText(frame, fps, (0, frame_height - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)

        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            print('已成功退出！将自动返回上一级')
            break
    camera.release()
    cv2.destroyAllWindows()


def local_recognize(img_path, database_path, min_rec_size):

    """
    识别出本地图片中的人物姓名或ID
    :param img_path: 被识别的图片的路径
    :param database_path: FaceRecognition_128D.json文件路径
    :param min_rec_size: 能被摄像头识别到的最小人脸大小
    """

    frame = cv2.imdecode(np.fromfile(img_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持文件名包含中文字符
    height = frame.shape[0]
    width = frame.shape[1]

    if height > 1500 or width > 2000:
        frame = cv2.resize(frame, (int(width / 2), int(height / 2)))

    rects, landmarks = face_detect.detect_face(frame, min_rec_size)  # 检测人脸时，能被检测到的最小人脸大小为32*32
    aligns = []
    positions = []

    for (i, rect) in enumerate(rects):
        aligned_face, face_pos = aligner.align(128, frame, landmarks[i])
        aligns.append(aligned_face)
        positions.append(face_pos)
        features_arr = extract_feature.get_features(aligns)
        recog_data = identify_people(database_path, features_arr, positions, thres=THRES, percent_thres=PERCENT_THRES)

    for (i, rect) in enumerate(rects):
        cv2_im = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # cv2和PIL中颜色的hex码的储存顺序不同
        pil_im = Image.fromarray(cv2_im)
        draw = ImageDraw.Draw(pil_im)
        font = ImageFont.truetype('simsun.ttc', 25)  # simsun.ttc为宋体，如果系统没有该字体，请自行添加,字体大小25
        draw.text((rect[0], rect[1]), recog_data[i][0] + ' - ' + str(recog_data[i][1])[0:4] + '%',
                  (255, 255, 255), font=font)  # 支持中文字符，取结果字符串的前3位
        frame = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)

        if recog_data[i][1] < 70:
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)

        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), color)  # 框选出检测出的人脸

    print('按q键退出')
    while True:
        cv2.imshow('Result', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按q键退出
            print('已成功退出！将自动返回上一级')
            break
    cv2.destroyAllWindows()


"""
FaceRecognition_128D.json文件的数据结构：
{
"ID":
    {
        "Center"：[[128D vector]],
        "Right"：[[128D vector]],
        "Letf"：[[128D vector]]
    }
}
"""


def identify_people(database_path, features_arr, positions_arr, thres, percent_thres):

    """
    :param database_path: FaceRecognition_128D.json文件路径
    :param features_arr: 屏幕上被检测出的所有人脸的128D Features列表
    :param positions_arr: 屏幕上被检测出的所有人脸的脸部位置类型
    :param thres: 距离阈值，默认0.6
    :param percent_thres: 相似度百分比阈值，默认70%
    :return returnRes: 识别的结果，相似度百分比
    """

    if not os.path.isfile(database_path):
        print('未找到' + DB_PATH + '文件,请检查后重试！')
    with open(database_path, 'r') as f:
        data_set = json.loads(f.read())

    return_res = []
    for i, features_128D in enumerate(features_arr):
        result = '未知'
        smallest = sys.maxsize
        for person in data_set.keys():
            person_data = data_set[person][positions_arr[i]]
            for data in person_data:
                distance = np.sqrt(np.sum(np.square(data-features_128D)))
                if distance < smallest:
                    smallest = distance
                    result = person
        percentage = min(100, 100 * thres / smallest)
        if percentage <= percent_thres:
            result = '未知'
        return_res.append((result, percentage))
    return return_res


"""
导入人脸特征的原理：
    - >用户输入他/她的名字或ID - >从摄像头中捕捉图像 - >检测面部 - >裁剪并对齐
    - >脸部然后分为三种类型：Left,Right,Center
    - >提取128D vectors （人脸特征）
    - >将每个新提取的人脸128D vectors附加到其对应的位置(Center,Right,Left)
    - >按q键停止捕捉
    - >找出每个类别中128D vectors的中心
    - >保存到FaceRecognition_128D.json文件
"""


def create_cam_data(database_path, min_face_size):

    """
    捕获摄像头图像，生成并导入新的人脸特征
    :param database_path: FaceRecognition_128D.json文件路径
    :param min_face_size: 能被检测到的最小人脸大小
    """

    print('请输入新用户的名字或ID：')
    new_name = input()

    if not os.path.isfile(database_path):
        print('运行目录未找到' + DB_PATH + '文件，将自动生成！')
        with open(database_path, 'w') as f:
            f.write('{}')

    with open(database_path, 'r') as f:
        if f.read() == '':  # 如果FaceRecognition_128D.json文件为空，则写入｛｝，避免json解析出错
            with open(database_path, 'w') as f:
                f.write('{}')
    with open(database_path, 'r') as f:
        data_set = json.loads(f.read())

    person_imgs = {'Left': [], 'Right': [], 'Center': []}
    person_features = {'Left': [], 'Right': [], 'Center': []}
    print('请缓慢转动头部，低头和抬头，确保获取全面的面部特征。按q键退出并保存。')
    camera = cv2.VideoCapture(0)  # 启动摄像头，获取图像

    while True:
        _, frame = camera.read()
        rects, landmarks = face_detect.detect_face(frame, min_face_size)  # 能被检测到的最小人脸大小

        for (i, rect) in enumerate(rects):
            aligned_frame, pos = aligner.align(128, frame, landmarks[i])
            person_imgs[pos].append(aligned_frame)
            cv2.imshow('Acquire Facial Features', aligned_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下q键退出
            break
    camera.release()
    cv2.destroyAllWindows()

    for pos in person_imgs:
        person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]

    data_set[new_name] = person_features
    with open(database_path, 'w') as f:
        f.write(json.dumps(data_set))
    print('添加人脸特征成功！将自动返回上一级')


def create_lcl_data(database_path,
                    faces_dir,
                    min_face_size):

    """
    读取本地人像图片，生成并导入新的人脸特征
    :param database_path: FaceRecognition_128D.json文件路径
    :param faces_dir: 本地人脸图片的目录
    :param min_face_size: 能被检测到的最小人脸大小
    """

    if not os.path.exists(faces_dir):  # 判断faces_dir是否存在，不存在就退出
        print('%s 文件夹不存在，请检查后重试' % faces_dir)

    if not os.path.isfile(database_path):
        print('运行目录未找到' + DB_PATH + '文件，将自动生成！')
        with open(database_path, 'w') as f:
            f.write('{}')

    with open(database_path, 'r') as f:
        if f.read() == '':  # 如果FaceRecognition_128D.json文件为空，则写入｛｝，避免json解析出错
            with open(database_path, 'w') as f:
                f.write('{}')
    with open(database_path, 'r') as f:
        data_set = json.loads(f.read())

    face_name = []
    face_path = []
    error_name = []

    for img_path in os.listdir(faces_dir):
        if (
            img_path.endswith('jpg')
            or img_path.endswith('jpeg')
            or img_path.endswith('png')
            or img_path.endswith('bmp')
        ):
            face_name.append(os.path.splitext(img_path)[0])
            face_path.append(faces_dir + img_path)

    for new_name, new_path in zip(face_name, face_path):
        print('正在导入 ' + new_name)
        frame = cv2.imdecode(np.fromfile(new_path, dtype=np.uint8), cv2.IMREAD_COLOR)  # 支持文件名包含中文字符
        rects, landmarks = face_detect.detect_face(frame, min_face_size)  # 能被检测到的最小人脸大小

        if len(rects) == 0:
            error_name.append(new_name)
        else:
            for (i, rect) in enumerate(rects):
                aligned_frame, pos = aligner.align(128, frame, landmarks[i])
                person_imgs = {'Left': [], 'Right': [], 'Center': []}
                person_features = {'Left': [], 'Right': [], 'Center': []}
                person_imgs[pos].append(aligned_frame)
                cv2.imshow('Acquire facial features', aligned_frame)

                for pos in person_imgs:
                    person_features[pos] = [np.mean(extract_feature.get_features(person_imgs[pos]), axis=0).tolist()]
                data_set[new_name] = person_features

        if cv2.waitKey(1) & 0xFF == ord('q'):  # 按下q键退出
            break

    with open(database_path, 'w') as f:
        f.write(json.dumps(data_set))

    cv2.destroyAllWindows()
    print('导入人脸特征完毕！\n其中\n')
    error_text = ''
    for error in error_name:
        print(error)
        error_text = error_text + error + '\n'

    with open(ERROR_PATH, 'w') as f:  # 将未导入的人脸名字写出到error_people文本中
        f.write(error_text)

    print('导入失败，请检查后重试或更换更清晰的图片！将自动返回上一级')


def delete_people(database_path):

    """删除FaceRecognition_128D.json文件中的指定的人脸特征"""

    if not os.path.isfile(database_path):
        print('运行目录未找到' + DB_PATH + '文件，请检查后重试！')
    else:
        with open(database_path, 'r') as f:
            data_set = json.loads(f.read())
        i = 0
        iscycle = True
        list_data = list(data_set.keys())

        while iscycle:
            print('以下是数据库中包含的人脸特征：')
            for person in list_data:
                i = i + 1
                print(i, person)

            print(
                '\n返回上一级请输入：0'
                + '\n删除所有人脸特征请输入：-1'
                  )

            lcl_temp = input('请输入想要删除的人的序号:')
            if lcl_temp == '':
                print('不能选择空，请重新选择操作！\n')
            else:
                lpick = int(lcl_temp)
                if lpick == -1:
                    iscycle = False
                    with open(database_path, 'w') as f:
                        f.write('{}')
                    print('\n已删除所有人脸特征\n')
                elif lpick == 0:
                    iscycle = False
                    i = 0
                elif lpick < 1 or lpick > len(list_data):
                    print('序号错误，请检查后重新输入\n')
                    i = 0
                else:
                    iscycle = False
                    for index, person in enumerate(list_data):
                        if person == list_data[lpick - 1]:
                            del data_set[person]
                            print('已删除' + list_data[lpick - 1] + '\n')

                    # print(data_set)
                    with open(database_path, 'w') as f:
                        f.write(json.dumps(data_set))


def read_cfg(cfg_file, key):

    """
    读取本地的配置信息
    :param cfg_file: 配置文件路径
    :param key: 键
    :return:值
    """

    cfg_data = json.loads(open(cfg_file).read())
    return cfg_data[key]


class FaceRecognizeGraph(object):
    def __init__(self):

        """ 创建graph属性"""

        self.graph = tf.Graph()


isload = True
while True:
    print(
        '\n1.通过摄像头导入人脸特征\n'
        + '2.获取本地图片，导入人脸特征\n'
        + '3.识别摄像头中的人物姓名或ID\n'
        + '4.识别本地图片中的人脸\n'
        + '5.删除数据库中的人脸特征\n'
        + '0.退出'
    )

    CFG_PATH = './config/config.json'
    MODEL_DIR = read_cfg(CFG_PATH, 'model_dir')
    DB_PATH = read_cfg(CFG_PATH, 'db_path')
    FACES_DIR = read_cfg(CFG_PATH, 'faces_dir')
    MIN_RECOGNIZE_SIZE = read_cfg(CFG_PATH, 'min_recognize_size')
    MIN_CREATE_SIZE = read_cfg(CFG_PATH, 'min_create_size')
    THRES = read_cfg(CFG_PATH, 'thres')
    PERCENT_THRES = read_cfg(CFG_PATH, 'percent_thres')
    ERROR_PATH = read_cfg(CFG_PATH, 'error_path')

    # print(model_dir, db_path)
    temp = input('请选择操作:')
    if temp == '':
        print('不能选择空，请重新选择操作！\n')
    else:
        pick = int(temp)
        if pick == 1:
            if isload:
                isload = False
                face_rec_graph = FaceRecognizeGraph()
                aligner = AlignCustom()
                extract_feature = FaceFeature(face_rec_graph, MODEL_DIR)
                face_detect = MTCNNDetect(face_rec_graph, 2, MODEL_DIR)
                print('MTCNNDetect：' + str(face_detect))
            create_cam_data(DB_PATH, MIN_CREATE_SIZE)
        elif pick == 2:
            if isload:
                isload = False
                face_rec_graph = FaceRecognizeGraph()
                aligner = AlignCustom()
                extract_feature = FaceFeature(face_rec_graph, MODEL_DIR)
                face_detect = MTCNNDetect(face_rec_graph, 2, MODEL_DIR)
                print('MTCNNDetect：' + str(face_detect))
            create_lcl_data(DB_PATH, FACES_DIR, MIN_CREATE_SIZE)
        elif pick == 3:
            if isload:
                isload = False
                face_rec_graph = FaceRecognizeGraph()
                aligner = AlignCustom()
                extract_feature = FaceFeature(face_rec_graph, MODEL_DIR)
                face_detect = MTCNNDetect(face_rec_graph, 2, MODEL_DIR)
                print('MTCNNDetect：' + str(face_detect))
            camera_recognize(DB_PATH, MIN_RECOGNIZE_SIZE)
        elif pick == 4:
            face_path = input('请输入欲识别的图片的路径:')
            if isload:
                isload = False
                face_rec_graph = FaceRecognizeGraph()
                aligner = AlignCustom()
                extract_feature = FaceFeature(face_rec_graph, MODEL_DIR)
                face_detect = MTCNNDetect(face_rec_graph, 2, MODEL_DIR)
                print('MTCNNDetect：' + str(face_detect))
            local_recognize(face_path, DB_PATH, MIN_RECOGNIZE_SIZE)
        elif pick == 5:
            delete_people(DB_PATH)
        elif pick == 0:
            print('正在退出...')
            gc.collect()
            sys.exit(0)
        else:
            print('非法输入！请检查后重新输入\n')

