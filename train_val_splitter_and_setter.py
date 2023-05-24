# The code generates validation.txt and train.txt files. Seperation of images is done randomly. You can specify val-train ratio.
# So you can generate validation and train datasets for training an Artificial Neural Network

# source : https://github.com/4lparslan

import os
import cv2
import random

# Specify validation dataset ratio. %20 val %80 train recommended.
VAL_RATIO = 20
TEST_RATIO = 10

# specify the dataset path
path = r'dataset'

train = open('data/train.txt','w')
val = open('data/valid.txt','w')
test = open('data/test.txt','w')

all_images = []

for filename in os.listdir(path):
    if filename.endswith(".jpeg") or filename.endswith(".jpg"):
        fullname = os.path.join(path, filename)
        all_images.append(fullname)
        
random.shuffle(all_images)

length = len(all_images)
val_size = int(length * VAL_RATIO / 100)
test_size = int(length* TEST_RATIO / 100)
train_size = length - (val_size + test_size)

for i in range(val_size):
    val.write(all_images[i])
    val.write('\n')

for k in range(test_size):
    test.write(all_images[k])
    test.write('\n')

for j in range(train_size):
    train.write(all_images[j+val_size+test_size])
    train.write('\n')
    
train.close()
val.close()
test.close()