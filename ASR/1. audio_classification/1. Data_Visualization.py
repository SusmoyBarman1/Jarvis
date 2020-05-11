# -*- coding: utf-8 -*-
"""
Created on Sun May 10 21:47:07 2020

@author: Susmoy
"""
import os
import glob
import numpy as np
from scipy.io import wavfile
import matplotlib.pyplot as plt

main_dir = "C://Users//Susmoy//Desktop//audio_classification"
dataDir = os.path.join(main_dir, "wavfiles")

classDist = []

means = []
data = []
classes = []


for dirs in os.listdir(dataDir):
    
    count = 0
    for i in os.listdir(os.path.join(dataDir, dirs)):
        path = os.path.join(dataDir, dirs)
        path = os.path.join(path, i)
        #print(path)
        rate, signal = wavfile.read(path)
        length = signal.shape[0]/rate
        count = count + length
        data.append([dirs, i, length])
    
    count = count/30
    means.append(count)
    classes.append(dirs)
        
    
    

#print(data)
#for i in range(len(data)):
    #print(data[i])

print(len(means))



fig, ax = plt.subplots()
ax.set_title("Class Distribution", y=1.08)
ax.pie(means, labels=classes, autopct='%1.1f%%', shadow=False, startangle=90)
ax.axis('equal')
plt.show()





