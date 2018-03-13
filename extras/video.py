
# coding: utf-8

# In[23]:


import cv2
import tqdm
import numpy as np
import os
import time
from random import shuffle
input_loc="/home/ghost/Downloads/Transistos.mp4"
IMG_SIZE1=175
IMG_SIZE2=150
testing_data = []
time_start=time.time()
cap = cv2.VideoCapture(input_loc)
video_length = cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1
print ("Number of frames: ", video_length)
empty_frames = 0
valid_frames = 0
count = 0
print ("Converting video..\n")
# Start converting the video
while cap.isOpened():
    # Extract the frame
    ret, frame = cap.read()
    # Write the results back to output location.
    if (count % 1 == 0 and ret):
        valid_frames = valid_frames + 1
        img = cv2.resize(frame, (IMG_SIZE1,IMG_SIZE2))
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        img_num1 = "%#05d" % (count+1)
        # print(img_num1)
	#Now appending frame to the numpy array
        testing_data.append([np.array(img),img_num1])
    if(ret == False):
        empty_frames = empty_frames + 1
    count = count + 1
    # If there are no more frames left
    if ((count > (video_length-1)) or (empty_frames>1)):
        # Log the time again
        time_end = time.time()
        # Release the feed
        cap.release()
        # Print stats
        print ("Done extracting frames.\n%d frames found" % count)
        print ("It took %d seconds forconversion." % (time_end-time_start))
        print("Total Valid Frames : %d " % (valid_frames))
        print("Total Empty Frames : %d " % (empty_frames))
        break
        
#saving the file as numpy array 
np.save('/home/ghost/Desktop/test_data1.npy', testing_data)

