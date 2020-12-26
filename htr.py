#Importing required libraries, i.e. OpenCV, Numpy and Tensor Flow
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#importing the dataset form mnist
mnist=tf.keras.datasets.mnist
#splitting the data in training and testing datasets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#scaling down the training and test datasets
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

