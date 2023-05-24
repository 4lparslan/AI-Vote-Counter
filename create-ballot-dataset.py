import cv2 as cv
from PIL import Image, ImageOps
import random
import os

STEP = 30

# Get base image for generating others
img = Image.open("ballot-bw.png")

# Get seal image
seal = Image.open("seal.png")
seal_size = 110
seal = seal.resize((seal_size,seal_size))
# randomize the rotating process
rotated = seal.rotate(random.randint(0,360))

# Determine working area getting sizes
base_w, base_h = img.size
seal_w, seal_h = seal.size

outputFolder = "dataset"

if not os.path.exists(outputFolder): #check if there is an output folder. If not create it
  os.makedirs(outputFolder)

counter = 0
w = 0
h = 0

# annotation data format
# <object-class> <x> <y> <width> <height>
object_class = 0


for w in range(0,base_w - seal_size, STEP):
    for h in range(0,base_h - seal_size, STEP):
        rotated = seal.rotate(random.randint(0, 360))
        img_copy = img.copy()
        img_copy.paste(rotated, (w,h), rotated)

        # save image
        img_copy.save(outputFolder + "/" + str(counter) + ".jpeg", format="jpeg")

        # save annotations in YOLO format
        f = open(outputFolder + "/" + str(counter) + ".txt", "x")
        center_x = w+(seal_size/2)
        center_y = h+(seal_size/2)
        f.write(str(object_class)+ " " + str(center_x/base_w) + " " + str(center_y/base_h)
                + " " + str(seal_size/base_w) + " " + str(seal_size/base_h))
        f.close()


        print(counter)
        counter+=1


