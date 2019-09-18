# import libraries
import numpy as np
import pandas as pd
from input_data import readJSON, downloadTraining
from sanitizer import processing, plot
import cv2

Plates = []
Sanity_list = []
temp_img = []



# img_ori = cv2.imread('test_plate.jpg')
# height, width, channel = img_ori.shape
# print(img_ori.shape)
# RGB_img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
# height, width, channel = RGB_img.shape
# gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)    # grayscaled image
# temp_img.append(gray)

data = readJSON()
Plates = downloadTraining(data)
print(type(Plates[0]))
Sanity_list = processing(Plates)
plot(Sanity_list)