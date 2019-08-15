This part mainly attempts to use the deep learning methods to directly classify facial expressions:

Previously, there have been related research. On the test data set published by fer2013, the average accuracy of 7 common expressions is about 0.73,
(such as: https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch)

However, subsequent research shows that the expression dataset provided by the original data fer2013 is marked incorrectly. Therefore,
relevant research is re-labeled to obtain the FERPlus. The vgg13-based deep learning model designed by Zhang Zhengyou was applied to the dataset and learned in a variety of modes. The average accuracy of the seven expressions finally obtained was about 0.83.(https://arxiv.org/abs/1608.01041)

Based on the RAF-DB dataset【Paper: Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild】, it is a facial expression dataset closer to the natural scene.The expression recognition model based on the dataset can better adapt to the natural scene.
Some recent studies have used the fast rcnn model to get better accuracy on this data set,(Paper: Facial Expression Recognition with Faster R-CNN), In this paper, the avg accuracy was about 0.83.

If you are not familiar with faster rcnn, you can learn through the relevant blogs(https://blog.csdn.net/u011534057/article/details/51247371).

【Experimental descriptions】
For some reasons, the source code may not be shared for the time being. The following will share some experimental processes and instructions；

【database】: RAF-DB and CK

【method】 faster rcnn
In order to facilitate comparison with existing papers on the RAF-DB dataset, the same test set is used in the paper;
The training set selects the RAF-DB and CK expression data. Because the samples of each category are not balanced,
the data enhancement method is adopted, including the upsampling operation, adjusting the brightness and channel color, and rotating;

【baseline】
The weight of the original vgg16 model is trained through the ImageNet dataset, but the ImageNet dataset differs greatly from the existing facial expression data architecture, so it is not appropriate to directly select the original vgg16 model weight.

The vgg face is often used to study the face recognition model. The base model uses the vgg16 model and then fine-tunes it on the face recognition dataset, and the vgg face model weights can be finetuned in the RAF dataset,continue to fine-tune around 400 epoches in the category of expressions; get the final baseline weight;

【faster rcnn】
Using the faster rcnn to load the baseline weights, the expression recognition is performed as the target detection task,
the face border detection uses the MTCNN model, and the data set is re-formed into the VOC data set format.and the model is trained on the RAF-DB dataset, and finally the average classification accuracy in the seven categories increased to 0.88.


【the other implementation method】
The vgg16 model is often used to solve category classification. However, when vgg16 is used as the basis weight and the model is fine-tuned, the average accuracy of the seven expressions on the RAF-DB dataset can only reach 0.78-0.8.

1.In order to try to solve the expression classification problem by using this type of method, the MTCNN model is used to detect the face frame, and the detected face region and the corresponding expression are used as training data, and for the baseline weight, the vgg face model weight is used.
2.At the same time, the calculation method of the loss function is optimized. In the classification problem, the softmax layer is often used to judge the final classification probability. However, in the expression classification, there are certain differences between the same expression categories, and the pictures are also different. Expression classification is performed using a DLP-CNN based network.(Paper: Reliable Crowdsourcing and Deep Locality-Preserving Learning for Expression Recognition in the Wild)

【Mobile implementation】（官方：https://github.com/tensorflow/tensorflow/tree/master/tensorflow/examples/android）
Regarding the implementation of the mobile terminal, on the basic data set RAF-DB, the expression recognition model is trained using mobilenet-ssd, and combined with the official Android implementation of tensorflow, and finally transplanted to the mobile terminal;
