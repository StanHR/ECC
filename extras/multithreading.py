
# coding: utf-8

# In[1]:


import threading
import cv2
from threading import Thread
import math
import time
import numpy as np
import os, random
import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected, flatten
from tflearn.layers.estimator import regression
import tensorflow as tf

# In[2]:

class videoProcessing:
    def __init__(self,path):
        self.path = path
        
    def videoFetch(self,thread_id):
        path = self.path
        img_width = 175
        img_height = 150
        testing_data = []
        start_time = time.time()
        cap = cv2.VideoCapture(path)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
        fragment_size = frame_count/8
        init_frame = math.floor(fragment_size*thread_id)
        end_frame = math.floor(fragment_size*(thread_id+1)-1)
        count = init_frame
        cap.set(1,init_frame)
        while cap.isOpened():
            ret, frame = cap.read()
            if(ret):
                img = cv2.resize(frame, (img_width,img_height))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img_num = "%#05d" % (count+1)
                testing_data.append([np.array(img),img_num])
            count = count+1
            if (count == end_frame):
                end_time = time.time()
                cap.release()
                print ("{} Done extracting frames.\n{} frames found".format(thread_id,end_frame-init_frame))
                print ("It took %d seconds forconversion." % (end_time-start_time))
                break
        # np.save('/home/ghost/Desktop/ecc/test_data_{}.npy'.format(thread_id), testing_data)
        LR = 0.0001
        MODEL_NAME = 'ECR-{}-{}.model'.format(LR, '2conv-basic')
        tf.reset_default_graph()
        convnet = input_data(shape=[None, IMG_SIZE1, IMG_SIZE2, 1], name='input')
		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 32, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = dropout(convnet, 0.2)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 64, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)
		convnet = conv_2d(convnet, 128, 2, activation='relu')
		convnet = max_pool_2d(convnet, 2)

		convnet = flatten(convnet)

		convnet = fully_connected(convnet, 1024, activation='relu')
		convnet = dropout(convnet, 0.4)

		convnet = fully_connected(convnet, 2, activation='softmax')
		convnet = regression(convnet, optimizer='adam', learning_rate=LR, loss='binary_crossentropy', name='targets')

		model = tflearn.DNN(convnet, tensorboard_dir='log')



		if os.path.exists('{}.meta'.format(MODEL_NAME)):
		    model.load('../'+MODEL_NAME)
		    print('model loaded!')



# In[3]:


obj1 = videoProcessing("/home/ghost/Downloads/Transistos.mp4")


# In[4]:


vF_thread0 = Thread(target=obj1.videoFetch,args=(0,))
vF_thread1 = Thread(target=obj1.videoFetch,args=(1,))
vF_thread2 = Thread(target=obj1.videoFetch,args=(2,))
vF_thread3 = Thread(target=obj1.videoFetch,args=(3,))
vF_thread4 = Thread(target=obj1.videoFetch,args=(4,))
vF_thread5 = Thread(target=obj1.videoFetch,args=(5,))
vF_thread6 = Thread(target=obj1.videoFetch,args=(6,))
vF_thread7 = Thread(target=obj1.videoFetch,args=(7,))

# In[5]:


vF_thread0.start()
vF_thread1.start()
vF_thread2.start()
vF_thread3.start()
vF_thread4.start()
vF_thread5.start()
vF_thread6.start()
vF_thread7.start()
