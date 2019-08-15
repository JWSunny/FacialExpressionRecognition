databases:
常见的人脸表情数据集： fer2013(kaggle竞赛)， CK和CK+数据集(http://www.consortium.ri.cmu.edu/ckagree/)， RAF-DB(http://www.whdeng.cn/raf/model1.html)
当然还有其他的表情数据：ISED、EmotiW(AFEW)【视频数据】

方法说明：
常见的表情识别方法：选择基础网络【vgg或者resnet等】，新增相关的卷积层或者池化层操作，在表情数据集上进行微调操作；
(为了排除图片中背景的影响，常常需要加入相关的人脸检测算法，选择人脸框作为深度学习模型的输入)
本篇尝试从人脸关键点的角度出发，研究与表情之间的关系；

关键点方法实验及说明：
之前的研究的两个关键点方法说明：https://github.com/JWSunny/Face_Landmarks（可参考）；

基于关键点的表情识别：
1.Exploiting Facial Landmarks for Emotion Recognition in the Wild
概论文主要研究关键点之间的特征，以68个点为例，主要最多是68*68组合特征；
（1）纹理特征：选取8个刻度值，12个方向，最终得到68*8*12个特征；
（2）几何特征：嘴和眼睛部位的偏心率特征、眼睛和眉毛、嘴和鼻子、上嘴唇和下嘴唇之间的距离；

2.EmotionalDAN  (CVPR 2018)  I Know How You Feel: Emotion Recognition with Facial Landmarks
基于DAN模型的关键点特征抽取，进行表情预测，论文中描述已经在国外进行商用；在CK+数据上能达到0.736【7种表情】，JAFFE[0.465], ISED【0.62】

实验说明：
数据集：CK表情数据集
关键点定位：采用的https://github.com/JWSunny/Face_Landmarks中基于DCNN的检测方法；
人脸检测：为了排除背景的影响，对于原图进行相关的人脸检测:
（https://github.com/Seymour-Lee/face-detection-ssd-mobilenet）
https://github.com/opencv/opencv/blob/24bed38c2b2c71d35f2e92aa66648f8485a70892/samples/dnn/resnet_ssd_face_python.py
裁剪的图片的处理：进行相关的仿射变换，使得人脸图片具有相同的尺寸，另外关键点也做对应的变换操作；

数据集的处理：
由于CK数据集中，样本是不均衡，对于样本进行随机的旋转操作，进行数据平衡数据（即采用上采样的操作，扩充少数样本, 每类大概1500张，偏差200张以内）

【特征选取】：
对于识别的人脸关键点，进行几何特征的提取【上嘴唇和下嘴唇、眉毛和眼眶、下额和鼻子、嘴巴和鼻子、眉毛和鼻子、眼睛和鼻子 之间的距离】，作为特征；
并加入关键点 到 所有关键点中心点的距离特征；
【对于分类器的选择】：
1.选择简单的机器学习算法，随机森林(RF)分类器；
【最终的实验效果】：
anger(0.72), disgust(0.66), fear(0.73), happy(0.84), neutral(0.89), sadness(0.83), surprise(0.96),  avg(0.8)

2.设计简单的深度学习网路【全连接构成】，同样在CK数据集上；
【特征选择】：
每个点到68关键点中心点在x轴方向上的距离（68维）；每个点与中心点在y轴上的距离（68维）；每个点到中心点的距离（68维）；每个点与中心点的角度关系值（68维）
【网络---4层全连接网络】
layer1：272节点（68*4）；layer2：512节点；layer3：256节点；layer4：7维
【7种类别及平均准确率】
anger(0.66), disgust(0.65), fear(0.67), happy(0.84), neutral(0.85), sadness(0.70), surprise(0.93),  avg(0.76)

DCNN_get_landmarks.py 中对特征提取方法进行了描述；之前的人脸关键点定位模型文件暂未分享，需自行实现之前关键点的定位方法；
deep_learning_method.py 是自行设计的简单深度学习模型，进行表情识别；


【后续】
1.基于关键点的特征表示，应用到fer2013的表情识别数据上，但效果不佳，7种表情的准确性在0.55-0.6左右；
2.由于fer2013数据集，直接是人脸图片，无需人脸检测；
3.后续张正友在研究表情识别证明，fer2013标注不准确，重新标注得到 FERPlus(FER+) Emotion FER database
【论文：https://arxiv.org/abs/1608.01041】
数据连接：https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data
文中，采用不同的模式，之前在fer2013上的准确度大概在73%， 新的重新标注的数据集 ,7种表情平均准确度达到 0.83-0.84之间；

【实验】
利用基于关键点的特征表示方法，在新的 FERPlus 较为准确的 FER+数据集上，得到7种表情平均准确度 0.75左右；

目前，fer2013上比较好的模型是基于vgg19，数据集公开测试集的准确率大概0.73左右；
参考（https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch）

【Feature Relative Works】
database: Real-world Affective Faces Database, a large-scale facial expression database;
【Processing】
Similarly, through the above-mentioned methos based on facial landmarks, the landmarks of the faces in the emotion images can be
obtained, and the feature representation methods based on facial landmarks used to acquire the features of corresponding emotion images,
finally, choose machine learning classification models or simple deeplearning models designed by yourself.