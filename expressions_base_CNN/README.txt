This part mainly attempts to use the deep learning methods to directly classify facial expressions:

Previously, there have been related research. On the test data set published by fer2013, the average accuracy of 7 common expressions is about 0.73,
(such as: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)

However, subsequent research shows that the expression dataset provided by the original data fer2013 is marked incorrectly. Therefore,
relevant research is re-labeled to obtain the FERPlus. The vgg13-based deep learning model designed by Zhang Zhengyou was applied to the data set
and learned in a variety of modes. The average accuracy of the seven expressions finally obtained was about 0.83.(https://arxiv.org/abs/1608.01041)

Based on the RAF-DB dataset【Paper: Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild】,
it is a facial expression dataset closer to the natural scene.The expression recognition model based on the dataset can better adapt to the natural scene.
Some recent studies have used the fast rcnn model to get better accuracy on this data set,(Paper: Facial Expression Recognition with Faster R-CNN)
In this paper, the avg accuracy was about 0.83.

If you are not familiar with faster rcnn, you can learn through the relevant blogs(https://blog.csdn.net/u011534057/article/details/51247371).
