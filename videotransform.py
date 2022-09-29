import cv2, sys, random
from aiortc import MediaStreamTrack, RTCPeerConnection, RTCSessionDescription
from av import VideoFrame
import threading
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from util import *
from darknet import Darknet
from preprocess import prep_image, inp_to_image
import pandas as pd
import random 
import argparse
import pickle as pkl
#Use the local version of darknet_video
#import darknet_video
def prep_image(img, inp_dim):
    """
    Prepare image for inputting to the neural network. 
    
    Returns a Variable 
    """

    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = cv2.resize(orig_im, (inp_dim, inp_dim))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x, img):
    classes = load_classes('data/coco.names')
    colors = pkl.load(open("pallete", "rb"))
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    cls = int(x[-1])
    label = "{0}".format(classes[cls])
    print(label)
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1);
    return img
def arg_parse():
    """
    Parse arguements to the detect module
    
    """
    
    
    parser = argparse.ArgumentParser(description='YOLO v3 Cam Demo')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.25)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--reso", dest = 'reso', help = 
                        "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed",
                        default = "160", type = str)
    return parser.parse_args()
def generate(frame):
   # grab global references to the lock variable
   global lock
   # initialize the video stream
   cfgfile = "cfg/yolov3-tiny.cfg"
   weightsfile = "yolov3-tiny.weights"
   num_classes = 80

   args = arg_parse()
   confidence = float(args.confidence)
   nms_thesh = float(args.nms_thresh)
   start = 0
   CUDA =False
    

    
    
   num_classes = 80
   bbox_attrs = 5 + num_classes
    
   model = Darknet(cfgfile)
   model.load_weights(weightsfile)
    
   model.net_info["height"] = args.reso
   inp_dim = int(model.net_info["height"])
    
          
   img, orig_im, dim = prep_image(frame, inp_dim)
   if CUDA:
        im_dim = im_dim.cuda()
        img = img.cuda()
        model.cuda()
   model.eval()

   output = model(Variable(img), CUDA)
   output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)          
   output[:,1:5] = torch.clamp(output[:,1:5], 0.0, float(inp_dim))/inp_dim
   output[:,[1,3]] *= frame.shape[1]
   output[:,[2,4]] *= frame.shape[0]
   results=list(map(lambda x: write(x, orig_im), output))
   return(results)

class VideoTransform(MediaStreamTrack):
    kind = "video"
    boxes = None

    def __init__(self, track, transform):
        super().__init__()
        self.track = track
        self.transform = transform
        self.busy = True

    async def recv(self):
        #Grab frame from stream
        frame = await self.track.recv()
        img = frame.to_ndarray(format="rgb24")

        if(random.randint(0,5) == 0):
            results=generate(img)

        #If we have any detections add them to frame

        #Convert frame back to stream
        new_frame = VideoFrame.from_ndarray(img,format="rgb24")
        new_frame.pts = frame.pts
        new_frame.time_base = frame.time_base
        return new_frame

