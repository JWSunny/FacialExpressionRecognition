# -*- coding: utf-8 -*-#
#-------------------------------------
# Name:         DCNN_get_landmarks
# Description:  
# Author:       sunjiawei
# Date:         2019/8/14
#-------------------------------------

####  主要通过关键点间的特征进行学习

import argparse
import os.path
import numpy as np
import tensorflow as tf
import cv2, glob, random
from sklearn.ensemble import RandomForestClassifier
import os
import math
from time import time
from sklearn.externals import joblib
import pickle
import face_detector as fd
import pts_tools as pt
import itertools as itool

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

''' 
1.主要分享关于关键点的几何特征抽取方法 
2.关于关键点检测的pb文件模型，未进行分享；
3.关于表情数据集太多，也暂时未进行分享，可自行下载相关的CK数据集；
'''

emotions = ["anger","disgust","fear","happy","neutral","sadness","surprise"]
pb_dir = "./saved_model"
def create_graph():
    with tf.gfile.FastGFile(os.path.join(pb_dir, 'frozen_inference_graph.pb'), 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        _ = tf.import_graph_def(graph_def, name='')

###  主要进行扣取人脸，并将人脸做方正处理 128 * 128 尺度
### 加载图片数据
def read_image(img_file):
    """Read the corsponding image."""
    if os.path.exists(img_file):
        img = cv2.imread(img_file)
    return img

#### 提取图片中的人脸
def extract_face(file):
    global new_img
    """Extract face area from image."""
    image = read_image(file)
    conf, raw_boxes = fd.get_facebox(image=image, threshold=0.9)
    # fd.draw_result(image, conf, raw_boxes)

    for box in raw_boxes:
        # Move box down.
        diff_height_width = (box[3] - box[1]) - (box[2] - box[0])
        offset_y = int(abs(diff_height_width / 2))
        box_moved = pt.move_box(box, [0, offset_y])

        # Make box square.
        facebox = pt.get_square_box(box_moved)

        face_image = image[facebox[1]:facebox[3], facebox[0]: facebox[2]]

        if face_image.shape[0] * face_image.shape[1] != 0:
            preview_img = face_image.copy()
            new_img = cv2.resize(preview_img, (128, 128))
    return new_img


def optimise_pts(x,y):
    subset=[]
    i=0
    nose_tip=np.asarray((np.mean(x[30:36]),np.mean(y[30:36])))
    leye=np.asarray((np.mean(x[36:42]),np.mean(y[36:42])))
    reye=np.asarray((np.mean(x[42:48]),np.mean(y[42:48])))
    scale_dist=np.mean([np.linalg.norm(nose_tip-leye),np.linalg.norm(nose_tip-reye)])
    for lip_x,lip_y in zip(itool.combinations(x[48:68],2),itool.combinations(y[48:68],2)):#171
        x0=lip_x[0]-lip_x[1]
        y0=lip_y[0]-lip_y[1]
        dist=np.linalg.norm([x0,y0])
        subset.append(dist)
    for leyebr_x,leyebr_y in zip(x[17:22],y[17:22]):#30
        for leye_x, leye_y in zip(x[36:42],y[36:42]):
            x1= leyebr_x - leye_x
            y1= leyebr_y - leye_y
            dist=np.linalg.norm([x1,y1])
            subset.append(dist)
    for reyebr_x,reyebr_y in zip(x[22:27],y[22:27]):#30
        for reye_x, reye_y in zip(x[42:48],y[42:48]):
            x2= reyebr_x - reye_x
            y2= reyebr_y - reye_y
            dist=np.linalg.norm([x2,y2])
            subset.append(dist)
    for jaw_x, jaw_y in zip(x[5:12],y[5:12]):#7
        x3 = jaw_x-nose_tip[0]
        y3 = jaw_y-nose_tip[1]
        dist = np.linalg.norm([x3,y3])
        subset.append(dist)
    for nl_x,nl_y in zip(x[48:60],y[48:60]):#12
        x4 = nl_x-nose_tip[0]
        y4 = nl_y-nose_tip[1]
        dist = np.linalg.norm([x4,y4])
        subset.append(dist)
    for leyebr_x,leyebr_y, reyebr_x,reyebr_y in zip(x[17:22],y[17:22],x[22:27],y[22:27]):#==255
        x5 = leyebr_x - nose_tip[0]
        y5 = leyebr_y - nose_tip[1]
        dist1=np.linalg.norm([x5,y5])
        x6 = reyebr_x - nose_tip[0]
        y6 = reyebr_y - nose_tip[1]
        dist2=np.linalg.norm([x6,y6])
        dist= (dist1+dist2)/2
        subset.append(dist)
    subset.append(0)
    subset=np.array(subset)
    subset = subset/scale_dist*100
    subset = subset.tolist()
    return subset


'''
几何特征1 描述：
上嘴唇和下嘴唇、眉毛和眼眶、下额和鼻子、嘴巴和鼻子、眉毛和鼻子、眼睛和鼻子 之间的距离特征
'''
def getlandmark_features(image_file):
    if not tf.gfile.Exists(image_file): tf.logging.fatal('File does not exist %s', image_file)
    # Extract face image.
    newimg = extract_face(image_file)
    image_array = np.array(newimg)

    ## using ***.pb predict 68 landmarks
    with tf.Session() as sess:
        logits_tensor = sess.graph.get_tensor_by_name('k2tfout_0:0')
        predictions = sess.run(logits_tensor, {'input_1:0': image_array})
        marks = np.array(predictions).flatten() * 64
        marks = np.reshape(marks, (-1, 2))

        xlist = []
        ylist = []
        for i in range(len(marks)):
            xlist.append(float(marks[i][0]))
            ylist.append(float(marks[i][1]))
        result_data = optimise_pts(xlist, ylist)
        return result_data

#####  利用关键点来获取人脸的表情特征
''' 
几何特征2 描述：
每个点到68关键点中心点在x轴方向上的距离（68维）；每个点与中心点在y轴上的距离（68维）；
每个点到中心点的距离（68维）；每个点与中心点的角度关系值（68维）
'''
def getlandmark_features(image_file):

    if not tf.gfile.Exists(image_file): tf.logging.fatal('File does not exist %s', image_file)

    # Extract face image.
    newimg = extract_face(image_file)
    image_array = np.array(newimg)

    ## using ***.pb predict 68 landmarks
    with tf.Session() as sess:
        logits_tensor = sess.graph.get_tensor_by_name('logits/BiasAdd:0')
        predictions = sess.run(logits_tensor, {'input_image_tensor:0': image_array})
        marks = np.array(predictions).flatten() * 128
        marks = np.reshape(marks, (-1, 2))

        xlist = []
        ylist = []
        for i in range(len(marks)):
            xlist.append(float(marks[i][0]))
            ylist.append(float(marks[i][1]))
        xmean = np.mean(xlist)  #### Get the mean of both axes to determine centre of gravity
        ymean = np.mean(ylist)
        xcentral = [(x - xmean) for x in xlist]  ##### get distance between each point and the central point in both axes
        ycentral = [(y - ymean) for y in ylist]

        if xlist[26] == xlist[29]:
            anglenose = 0
        else:
            anglenose = int(math.atan((ylist[26]-ylist[29])/(xlist[26]-xlist[29]))*180/math.pi)

        if anglenose < 0:
            anglenose += 90
        else:
            anglenose -= 90

        landmarks_vectorised = []
        for x, y, w, z in zip(xcentral, ycentral, xlist, ylist):
            landmarks_vectorised.append(w)
            landmarks_vectorised.append(z)
            meannp = np.asarray((ymean, xmean))
            coornp = np.asarray((z, w))
            dist = np.linalg.norm(coornp - meannp)
            anglerelative = (math.atan((z-ymean)/(w-xmean))*180/math.pi) - anglenose
            landmarks_vectorised.append(dist)
            landmarks_vectorised.append(anglerelative)

    return landmarks_vectorised

###  训练时，用于数据集的划分---80%用于训练，20%用于测试
def get_files(image_path, emotion):
    print(os.path.join(image_path, emotion))
    files = glob.glob(os.path.join(image_path, emotion + '/*'))
    print(len(files))

    random.shuffle(files)
    training = files[:int(len(files) * 0.8)]
    prediction = files[-int(len(files) * 0.2):]
    return training, prediction


#### 获取所有的独立测试集
def get_test_files(image_path, emotion):
    files = glob.glob(os.path.join(image_path, emotion + '/*'))
    return files


######  获取所有训练集合，分成训练和测试集，分别获取对应数据集的关键点和表情
def make_sets(image_path):
    train_data = []
    test_data = []
    nums = []
    for emotion in emotions:
        train_image, test_image = get_files(image_path, emotion)
        nums.append(len(test_image))
        for item in train_image:
            landmarks = getlandmark_features(item)
            labels = np.zeros(7)
            labels[emotions.index(emotion)] = 1
            train_data.append((landmarks, labels))

        for item in test_image:
            landmarks = getlandmark_features(item)
            labels = np.zeros(7)
            labels[emotions.index(emotion)] = 1
            test_data.append((landmarks, labels))

    return train_data, test_data, nums

####  获取所有独立测试集的关键点和表情标签
def make_test_sets(image_path):
    absolute_test_data = []
    for emotion in emotions:
        absolute_image = get_test_files(image_path, emotion)
        for item in absolute_image:
            landmarks = getlandmark_features(item)
            labels = np.zeros(7)
            labels[emotions.index(emotion)] = 1
            absolute_test_data.append((landmarks, labels))
    return absolute_test_data



