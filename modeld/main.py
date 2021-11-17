# !/usr/bin/env python3
from common.transformations.camera import transform_img, eon_intrinsics
from common.transformations.model import medmodel_intrinsics
from common.lanes_image_space import transform_points
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

import cv2 
from tensorflow.keras.models import load_model
from common.tools.lib.parser import parser
import cv2
import sys

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession


config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

############## onnx to keras ###########
from keras.models import load_model 
import onnx 
from onnx2keras import onnx_to_keras 

path_onnx_model ="/home/gauti/Desktop/supercombo.onnx"
onnx_model = onnx.load(path_onnx_model)

### convertor 
input_all = [node.name for node in onnx_model.graph.input]

# print(input_all)

keras_model = onnx_to_keras(onnx_model,['input_imgs', 'desire', 'traffic_convention', 'initial_state'])
#############################


camerafile = sys.argv[1]
supercombo = load_model('models/supercombo.keras')

MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

x_path = np.linspace(0, 33, 33)

cap = cv2.VideoCapture(camerafile)

imgs = []

for i in tqdm(range(1000)):
  ret, frame = cap.read()
  img_yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV_I420)
  imgs.append(img_yuv.reshape((874*3//2, 1164)))
 

def frames_to_tensor(frames):                                                                                               
  H = (frames.shape[1]*2)//3                                                                                                
  W = frames.shape[2]                                                                                                       
  in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)                                                      
                                                                                                                            
  in_img1[:, 0] = frames[:, 0:H:2, 0::2]                                                                                    
  in_img1[:, 1] = frames[:, 1:H:2, 0::2]                                                                                    
  in_img1[:, 2] = frames[:, 0:H:2, 1::2]                                                                                    
  in_img1[:, 3] = frames[:, 1:H:2, 1::2]                                                                                    
  in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2,W//2))                                                              
  in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2,W//2))
  return in_img1

imgs_med_model = np.zeros((len(imgs), 384, 512), dtype=np.uint8)
for i, img in tqdm(enumerate(imgs)):
  imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                    output_size=(512,256))
frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0


recurrent_state = np.zeros((1,512))
desire = np.zeros((1,8))
drive_convention= np.zeros((1,2))
cap = cv2.VideoCapture(camerafile)

def softplus(x):
  # fix numerical stability
  return np.log1p(np.exp(x))
  # return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)

def softmax(x):
  x = np.copy(x)
  axis = 1 if len(x.shape) > 1 else 0
  x -= np.max(x, axis=axis, keepdims=True)
  if x.dtype == np.float32 or x.dtype == np.float64:
    np.exp(x, out=x)
  else:
    x = np.exp(x)
  x /= np.sum(x, axis=axis, keepdims=True)
  return x
  
def reshape(array,dimensions):
  # function to reshape the output tensors for a readable format
  pass 




for i in tqdm(range(len(frame_tensors) - 1)):
  inputs = [np.vstack(frame_tensors[i:i+2])[None], desire,drive_convention,recurrent_state ]
  
  ## TO do :: make a dictionary and append the data in the loop somehow for .npy file dummy data
  outs = keras_model.predict(inputs)

  """
  INDEXING OF THE FOLLOWING OUTPUT VECTOR OF SIZE (1,6472)
  As already mentioned here:-- https://github.com/commaai/openpilot/tree/master/models

  """
  path_dict = {}
  path_plans =  outs[:,:4955]

  path1, path2, path3, path4, path5 = np.split(path_plans,5,axis = 1)
  path_dict["path_prob"] = []
  path_dict["path1"] = path1[:,:-1].reshape(2,33,15)
  path_dict["path2"] = path2[:,:-1].reshape(2,33,15)
  path_dict["path3"] = path3[:,:-1].reshape(2,33,15)
  path_dict["path4"] = path4[:,:-1].reshape(2,33,15)
  path_dict["path5"] = path5[:,:-1].reshape(2,33,15)
  path_dict["path_prob"].append(path1[:,-1]) 
  path_dict["path_prob"].append(path2[:,-1])
  path_dict["path_prob"].append(path3[:,-1])
  path_dict["path_prob"].append(path4[:,-1])
  path_dict["path_prob"].append(path5[:,-1])

  # print(path_dict["path1"].shape)
  # print(path_dict["path1"][0, :, 2])  


  # print("******************")

  lanelines = outs[:,4955:5483]
  lane_dict = {}
  oll, ll, rll, orl = np.split(lanelines,4,axis =1)
  lane_dict["oll"] = oll.reshape(2,33,2)
  lane_dict["ll"] = ll.reshape(2,33,2)
  lane_dict["rll"] = rll.reshape(2,33,2)
  lane_dict["orl"] = orl .reshape(2,33,2)
  
  print(lane_dict["ll"][0,:,:])
  # laneline_prob = outs[:,5483:5491]
  # # print(laneline_prob)
  # road_edges = outs[:,5491:5755] 
  
  # leads = outs[:,5755:5857]

  # lead_probabilites = outs[:,5857:5860]
  # # print(softplus(lead_probabilites))
  transform_points
  
  # meta = outs[:,5868:5948]

  # pose = outs[:,5948:5960]

  recurrent_state = outs[:,5960:6472]

  # print("**********************************")
  # if i ==10:
  #   break

#   parsed = parser(outs)
#   # Important to refeed the state
  # state = outs[-1]
#   pose = outs[-2]
  ret, frame = cap.read()
  frame = cv2.resize(frame, (640, 420))
  # Show raw camera image
  cv2.imshow("modeld", frame)
  # Clean plot for next frame
  plt.clf()
  plt.title("lanes and path")

  plt.plot(lane_dict["ll"][0,:,0],range(0,33), "b-", linewidth=1)
  plt.plot(lane_dict["rll"][0,:,0],range(0,33), "r-", linewidth=1)
  plt.plot(lane_dict["oll"][0,:,0],range(0,33), "m-", linewidth=1)
  plt.plot(lane_dict["orl"][0,:,0],range(0,33), "k-", linewidth=1)
#   # path = path cool isn't it ?
#   plt.plot(parsed["path"][0], range(0, 192), "g-", linewidth=1)
#   #print(np.array(pose[0,:3]).shape)
#   plt.scatter(pose[0,:3], range(3), c="y")
  
  # Needed to invert axis because standart left lane is positive and right lane is negative, so we flip the x axis
  plt.gca().invert_xaxis()
  plt.pause(0.001)
  if cv2.waitKey(10) & 0xFF == ord('q'):
        break

plt.show()
