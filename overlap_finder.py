 
# One Time Installation
# !pip install ultralytics

#Imports : 

import pandas as pd
import numpy as np
import torch
import argparse
import cv2
import glob
from text_detection import text_boxes
from helper_functions import get_iou_and_intersection

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", type=str,help="path to input image")
args = vars(ap.parse_args())

#load the yolov5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=False)


image  =cv2.imread(args["image"]) #read the image
results = model(image)   #passing the image to model

yolo_boxes=[]  # yolo output bounding box coordinates are saved here.
for i in range(len(results.crop())):
  bbox = [val.cpu().detach().numpy().astype("int")+0 for val in results.crop()[i]["box"]]  # Take Each bounding box predictions , extract coordinates and convert tensor to int value.
  yolo_boxes.append(bbox)


model_location = "/content/frozen_east_text_detection.pb" # EAST Model - Text Boxes model location
text_boxes = text_boxes(args["image"],model_location) # predict text bounging boxes with east model.


def overlap_detection(yolo_boxes,text_boxes):
  """ Function to calulcate intersecion over union and intersection area over each yolo box and textr box to help us determine
  if there is any overlap"""
  iou,inter=0,0
  for yolo_box in yolo_boxes:
    for text_box in text_boxes:
      iou_val,inter_val=get_iou_and_intersection(yolo_box,text_box)  #find intersection over union
      iou+=iou_val
      inter+=inter_val

  if inter > 1000 :
    print("**************** Overlap is there ************")

  else:
    print("*********** No Overlap *************")

overlap_detection(yolo_boxes,text_boxes)

