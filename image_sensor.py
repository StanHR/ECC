import numpy as np 
import cv2
import tensorflow as tf 
import os
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression

class imageProcessing:
	def __init__(self, path):
		self.path = path

	def imageFetch(self):
		image_path = self.path
		IMG_SIZE1 = 175
		IMG_SIZE2 = 150
		test_data = []
		image = cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
		test_image = cv2.resize(image, (IMG_SIZE1,IMG_SIZE2))
		test_data.append([np.array(test_image)])

		LR = 0.0001
		MODEL_NAME = 'ECR-{}-{}.model'.format(LR, '2conv-basic')
		convnet = input_data(shape=[None, IMG_SIZE1, IMG_SIZE2, 1], name='input')

		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = dropout(convnet, 0.3)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = dropout(convnet, 0.3)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = dropout(convnet, 0.3)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)

		convnet = flatten(convnet)

		convnet = fully_connected(convnet, 256, activation='relu')
		convnet = dropout(convnet, 0.3)
		convnet = fully_connected(convnet, 512, activation='relu')
		convnet = dropout(convnet, 0.3)
		convnet = fully_connected(convnet, 1024, activation='relu')

		convnet = fully_connected(convnet, 2, activation='softmax')
		convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='binary_crossentropy', name='targets')

		model = tflearn.DNN(convnet, tensorboard_dir='log')

		if os.path.exists('{}.meta'.format(MODEL_NAME)):
		    model.load(MODEL_NAME)
		    print('Explicit Content Sensor Loaded !')


		for num,data in enumerate(test_data[:]):
			test_img = data[0]
			test_img_reshaped = test_img.reshape(IMG_SIZE1,IMG_SIZE2,1)
			model_out = model.predict([test_img_reshaped])[0]
			if np.argmax(model_out) == 1: 
				str_label= 1
				# Non-Explicit Content
			else:
				str_label= 0
				# Explicit Content
			final_image = cv2.imread(image_path)
			cv2.namedWindow('Image Viewer',cv2.WINDOW_NORMAL)
			if str_label == 0:
				final_image = cv2.blur(final_image,(200,200))	
			cv2.imshow("Image Viewer",final_image)
			k = cv2.waitKey(0) & 0xFF
			if k == 27:
				cv2.destroyAllWindows()		

path = input("Enter Image Path : ")
imageProcessor = imageProcessing(path)
imageProcessor.imageFetch()