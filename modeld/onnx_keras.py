import keras
import tensorflow as tf
# from keras.models import load_model 
import onnx 
from onnx2keras import onnx_to_keras 
from common.transformations.camera import transform_img, eon_intrinsics
print(eon_intrinsics)
print(eon_intrinsics[1,2])
# path_onnx_model ="/home/gauti/Desktop/supercombo.onnx"
# onnx_model = onnx.load(path_onnx_model)

# ### convertor 
# input_all = [node.name for node in onnx_model.graph.input]

# # print(input_all)

# keras_model = onnx_to_keras(onnx_model,['input_imgs', 'desire', 'traffic_convention', 'initial_state'])

# keras_model.summary()
# model.save("./supercombo_keras")