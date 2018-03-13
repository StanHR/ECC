
# coding: utf-8

# In[1]:

import tqdm
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

explicit_frames = []

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
        print("Thread {} starting Frame Extraction from {}th frame. Please wait for sometime.".format(thread_id,init_frame))
        end_frame = math.floor(fragment_size*(thread_id+1)-1)
        count = init_frame
        cap.set(1,init_frame)
        print("Frame Extraction in Progress by Thread {}".format(thread_id))
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
                print ("Thread {} finished extracting frames.\n{} frames found by Thread {}".format(thread_id,end_frame-init_frame,thread_id))
                print ("It took {} seconds for Frame Extraction by Thread {}".format(end_time-start_time,thread_id))
                break
        # np.save('/home/ghost/Desktop/ecc/test_data_{}.npy'.format(thread_id), testing_data)
        IMG_SIZE1 = 175
        IMG_SIZE2 = 150
        LR = 0.0001
        MODEL_NAME = 'ECR-{}-{}.model'.format(LR, '2conv-basic')
        tf.reset_default_graph()
        convnet = input_data(shape=[None, IMG_SIZE1, IMG_SIZE2, 1], 
		                     name='input')
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
        convnet = regression(convnet, optimizer='adam', 
		                     learning_rate=LR, 
		                     loss='binary_crossentropy', 
		                     name='targets')
        model = tflearn.DNN(convnet, tensorboard_dir='log')
        if os.path.exists('{}.meta'.format(MODEL_NAME)):
        	model.load(MODEL_NAME)
        	print('Explicit Content Censor Loaded by Thread {}'.format(thread_id))
        explicit = 0
       	non_explicit = 0
        print('Video Censoring started by Thread {}'.format(thread_id))
        for num,data in enumerate(testing_data[:]):
            # explicit: [1,0]
            # normal: [0,1]
            img_data = data[0]
            img_no = data[1]
            data = img_data.reshape(IMG_SIZE1,IMG_SIZE2,1)
            model_out = model.predict([data])[0]
            actual_frame_num = init_frame + num
            if(np.argmax(model_out)==0):
            	explicit_frames.append(actual_frame_num)

# In[3]:

path_to_video = input("Please Enter the Path of Video : ")
file_name = path_to_video.split('/')[-1]
vcache_path = './vcache/'+file_name+'.npy'

if not os.path.exists(vcache_path):
	obj1 = videoProcessing(path_to_video)

	vF_thread0 = Thread(target=obj1.videoFetch,args=(0,))
	vF_thread1 = Thread(target=obj1.videoFetch,args=(1,))
	vF_thread2 = Thread(target=obj1.videoFetch,args=(2,))
	vF_thread3 = Thread(target=obj1.videoFetch,args=(3,))
	vF_thread4 = Thread(target=obj1.videoFetch,args=(4,))
	vF_thread5 = Thread(target=obj1.videoFetch,args=(5,))
	vF_thread6 = Thread(target=obj1.videoFetch,args=(6,))
	vF_thread7 = Thread(target=obj1.videoFetch,args=(7,))

	vF_thread0.start()
	vF_thread1.start()
	vF_thread2.start()
	vF_thread3.start()
	vF_thread4.start()
	vF_thread5.start()
	vF_thread6.start()
	vF_thread7.start()

	vF_thread0.join()
	vF_thread1.join()
	vF_thread2.join()
	vF_thread3.join()
	vF_thread4.join()
	vF_thread5.join()
	vF_thread6.join()
	vF_thread7.join()	

	explicit_frames.sort()
	print(len(explicit_frames))
	np.save('./vcache/'+file_name+'.npy',explicit_frames)
else:
	vcache_data = np.load(vcache_path)
	explicit_frames = vcache_data



count = 0
cap = cv2.VideoCapture(path_to_video)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
print(total_frames)
fps = cap.get(cv2.CAP_PROP_FPS)
while True:
	ret,frame = cap.read()
	if count in explicit_frames:
		new_frame = cv2.blur(frame,(500,500))
		cv2.imshow('Media Player',new_frame)
	else:
		cv2.imshow('Media Player',frame)
	count = count+1
	if cv2.waitKey(int(fps)) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()

#13251 frames pointed as Explicit of xxx.mp4