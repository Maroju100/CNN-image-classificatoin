# CNN-image-classification
Welcome to the CNN-image-classificatoin wiki!

The goal of the project is to explore deep learning principles based on CNN for image classification problem.

The project is divided into two parts:

● Part 1: Build and train a CNN model using TensorFlow API for image classification of CIFAR-10 dataset and explore various deep learning principles and optimization techniques for improving the image classification accuracy.

● Part 2: Use a pre trained Google inception v3 model for classifying images and compare the classification results with the model built in Part 1.

Dateset:CIFAR10

API Used:TensorFlow

Part1 Accuracy of CNN based image classification algorithm depends on its underlying architecture. A CNN architecure is a result of various units such as Number of convolution layers, Filter weights, stride, pooling technique, fully connnected networks.

Following factors were considered while developing the CNN model:

Number of layers- Accuracy of CNN based image classifier can increases with the increase in the number of layers but large number of layers may result in overfitting, leading to poor results Depth of the network- Accuracy of image classification increases with the depth of the network but with the increase in depth the computational complexity is also increase leading to greater time consumption during training. Kernel size of layer- Convolution Kernel size increase the computatioanal complexity of the network. Fragmentation- Using multiple smaller Kernels inplace single big kernel can lead to better accuracy and reduced computational complexity. Max Pooling- Increase in kernel size for max pooling can lead to decrease accuracy due to downsampling of the image. Stride- Low value of stride ensures that the image pixels are not lost during processing. Based on the initial trial and errors and insights from the previous models proposed by research community the following model was considered for the CNN based on image classification.
