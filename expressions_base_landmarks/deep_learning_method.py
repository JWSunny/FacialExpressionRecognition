# -*- coding: utf-8 -*-#
#-------------------------------------
# Name:         deep_learning_method
# Description:  
# Author:       sunjiawei
# Date:         2019/8/14
#-------------------------------------

import random
import numpy as np
import tensorflow as tf
import os
from tensorflow.python.framework import graph_util
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

''' 
设计的简单的神经网络，对于 DCNN_get_landmarks.py中的 几何特征2 来进行表情预测；
'''

class EmotionClassifier:

    def __init__(self, num_classes, save_path=''):
        """ Constructor for EmotionClassifier that builds placeholders and the learning model.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :param save_path: A file path for the session variables to be saved. If not set the session will not be saved.
        :type save_path: A file path.
        """
        self.x = tf.placeholder("float", [None, 272], name="input")
        self.y = tf.placeholder("float", [None, num_classes], name="label")
        # self.model = self.build_model(num_classes)
        self.model = self.build_netural_model(num_classes)
        self.save_path = save_path

    def build_model(self, num_classes):
        """ Builds the Neural model for the classifier.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :return: The Neural model for the system.
        :rtype: A TensorFlow model.
        """
        weights = {
            'h1': tf.Variable(tf.random_normal([272, 512])),
            'h2': tf.Variable(tf.random_normal([512, 256])),
            # 'h3': tf.Variable(tf.random_normal([1024, 512])),
            # 'h4': tf.Variable(tf.random_normal([512, 256])),
            'out': tf.Variable(tf.random_normal([256, num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([512])),
            'b2': tf.Variable(tf.random_normal([256])),
            # 'b3': tf.Variable(tf.random_normal([512])),
            # 'b4': tf.Variable(tf.random_normal([256])),
            'out': tf.Variable(tf.random_normal([num_classes]))
        }

        layer1 = tf.nn.relu(tf.add(tf.matmul(self.x, weights['h1']), biases['b1']))
        # layer1 = tf.nn.dropout(layer1, 0.5)
        layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, weights['h2']), biases['b2']))
        # layer2 = tf.nn.dropout(layer2, 0.5)
        # layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, weights['h3']), biases['b3']))
        # layer3 = tf.nn.dropout(layer3, 0.5)
        # layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, weights['h4']), biases['b4']))
        return tf.matmul(layer2, weights['out']) + biases['out']

    def build_netural_model(self, num_classes):
        """ Builds the Neural model for the classifier.
        :param num_classes: The number of different classifications.
        :type num_classes: int
        :return: The Neural model for the system.
        :rtype: A TensorFlow model.
        """

        layer1 = tf.layers.dense(inputs=self.x, units=272, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='layer1')
        layer2 = tf.layers.dense(inputs=layer1, units=512, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='layer2')
        layer3 = tf.layers.dense(inputs=layer2, units=256, activation=tf.nn.relu, kernel_initializer=tf.glorot_uniform_initializer(), name='layer3')
        out = tf.layers.dense(inputs=layer3, units=num_classes, kernel_initializer=tf.glorot_uniform_initializer(), name='out')
        print(out)
        return out


    def train(self, training_data, testing_data, nums, epochs=5000):
        """ Trains a classifier with inputted training and testing data for a number of epochs.
        :param training_data: A list of tuples used for training the classifier.
        :type training_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param testing_data: A list of tuples used for testing the classifier.
        :type testing_data: A list of tuples each containing a list of landmarks and a list of classifications.
        :param epochs: The number of cycles to train the classifier for. Default is 50000.
        :type epochs: int.
        """
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = self.model, labels = self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)
        init, saver = tf.global_variables_initializer(), tf.train.Saver()

        with tf.Session() as sess:
            sess.run(init)
            correct_prediction = tf.equal(tf.argmax(self.model, 1), tf.argmax(self.y, 1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

            for epoch in range(epochs):
                batch_x, batch_y = [m[0] for m in training_data], [n[1] for n in training_data]
                _, avg_cost = sess.run([optimizer, cost], feed_dict={self.x: batch_x, self.y: batch_y})
                if epoch % 200 == 0:
                    print ("Epoch", '%04d' % (epoch), "cost = ", "{:.9f}".format(avg_cost))
                    print("Eval accuracy:",accuracy.eval({self.x: [m[0] for m in testing_data], self.y: [n[1] for n in testing_data]}))

                    emotions = ["anger", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]
                    a = 0
                    for i in range(len(nums)):
                        print("Emotion %s , Accuracy: %.3f" % (emotions[i], accuracy.eval(
                            {self.x: [m[0] for m in testing_data[a:(a + nums[i])]],
                             self.y: [n[1] for n in testing_data[a:(a + nums[i])]]})))
                        a += nums[i]

                    print("----++++++----")

            print ("Optimization Finished!")
            saver.save(sess, self.save_path) if self.save_path != '' else ''

            #### 第二种形式转化pb文件
            output_graph = './emotion_model/netural_emotion_frozen_graph.pb'

            graph_def = tf.get_default_graph().as_graph_def()
            # 模型持久化，将变量值固定
            output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph_def, ["out/BiasAdd"])  ### 需要保存节点的名字(输出的节点)
            with tf.gfile.GFile(output_graph, "wb") as f:  # 保存模型
                f.write(output_graph_def.SerializeToString())  # 序列化输出
            print("%d ops in the final graph." % len(output_graph_def.node))

    def classify(self, data):
        """ Loads the pre-trained model and uses the input data to return a classification.
        :param data: The data that is to be classified.
        :type data: A list.
        :return: A classification.
        :rtype: int.
        """
        init, saver = tf.initialize_all_variables(), tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, self.save_path)
            classification = np.asarray(sess.run(self.model, feed_dict={self.x: data}))
            return np.unravel_index(classification.argmax(), classification.shape)