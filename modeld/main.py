# !/usr/bin/env python3
import onnxruntime
import sys
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from common.transformations.model import medmodel_intrinsics
from common.transformations.camera import transform_img, eon_intrinsics


MAX_DISTANCE = 140.
LANE_OFFSET = 1.8
MAX_REL_V = 10.

LEAD_X_SCALE = 10
LEAD_Y_SCALE = 10

X_IDXS = [
         0.    ,   0.1875,   0.75  ,   1.6875,   3.    ,   4.6875,
         6.75  ,   9.1875,  12.    ,  15.1875,  18.75  ,  22.6875,
        27.    ,  31.6875,  36.75  ,  42.1875,  48.    ,  54.1875,
        60.75  ,  67.6875,  75.    ,  82.6875,  90.75  ,  99.1875,
       108.    , 117.1875, 126.75  , 136.6875, 147.    , 157.6875,
       168.75  , 180.1875, 192.]


PLAN_MEAN = 0
PLAN_X = 0
PLAN_Y = 1
LANE_MEAN = 0
LANE_Y = 0


def frames_to_tensor(frames):
    H = (frames.shape[1]*2)//3
    W = frames.shape[2]
    in_img1 = np.zeros((frames.shape[0], 6, H//2, W//2), dtype=np.uint8)

    in_img1[:, 0] = frames[:, 0:H:2, 0::2]
    in_img1[:, 1] = frames[:, 1:H:2, 0::2]
    in_img1[:, 2] = frames[:, 0:H:2, 1::2]
    in_img1[:, 3] = frames[:, 1:H:2, 1::2]
    in_img1[:, 4] = frames[:, H:H+H//4].reshape((-1, H//2, W//2))
    in_img1[:, 5] = frames[:, H+H//4:H+H//2].reshape((-1, H//2, W//2))
    return in_img1


def softplus(x):
    # fix numerical stability
    return np.log1p(np.exp(x))
    # return np.log1p(np.exp(-np.abs(x))) + np.maximum(x,0)


def load_model(path="models/supercombo.onnx"):
    return onnxruntime.InferenceSession(path, None)


def load_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frames.append(frame)

    return frames


def bgr_to_yuv(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    img_yuv = img_yuv.reshape((874*3//2, 1164)) #TODO: why are we doing this?
    return img_yuv


def transform_frames(frames):
    imgs_med_model = np.zeros((len(frames), 384, 512), dtype=np.uint8)
    for i, img in tqdm(enumerate(frames)):
        imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                        output_size=(512, 256))
    frame_tensors = frames_to_tensor(np.array(imgs_med_model)).astype(np.float32)/128.0 - 1.0
    return frame_tensors


def get_initial_inputs():
    recurrent_state = np.zeros((1, 512), dtype=np.float32)
    desire = np.zeros((1, 8), dtype=np.float32)
    drive_convention = np.zeros((1, 2), dtype=np.float32)
    return recurrent_state, desire, drive_convention


def forward_model(model, input_imgs, recurrent_state, desire, drive_convention):
    inputs = {'input_imgs': input_imgs,
              'initial_state': recurrent_state,
              'desire': desire,
              'traffic_convention': drive_convention
              }

    outs = model.run(None, inputs)[0]
    recurrent_state = outs[:, 5960:6472]

    assert recurrent_state.shape == (1, 512)

    return outs, recurrent_state


if __name__ == "__main__":
    input_video = sys.argv[1]  # '.../fcamera.hevc'
    model = load_model()

    bgr_frames = load_frames(input_video)
    yuv_frames = [bgr_to_yuv(frame) for frame in bgr_frames]
    prepared_frames = transform_frames(yuv_frames)

    recurrent_state, desire, drive_convention = get_initial_inputs()

    for i in tqdm(range(len(prepared_frames) - 1)):
        stacked_frames = np.vstack(prepared_frames[i:i+2])[None]
        assert stacked_frames.shape == (1, 12, 128, 256)

        outs, recurrent_state = forward_model(model, stacked_frames, recurrent_state, desire, drive_convention)

        """
        INDEXING OF THE FOLLOWING OUTPUT VECTOR OF SIZE (1,6472)
        As already mentioned here:-- https://github.com/commaai/openpilot/tree/master/models

        """
        path_plans = outs[:, :4955]

        paths = np.array(np.split(path_plans, 5, axis=1))
        paths = paths.squeeze() # (5, 991)

        best_idx = np.argmax(paths[:, -1], axis=0)
        best_path = paths[best_idx, :-1].reshape(2, 33, 15)

        print('logprobs:', paths[:, -1])
        print('best path', best_idx)
        print('furthest distance (any path):', np.max(paths))
        print('furthest distance (best path):', best_path[0, -1, 0])

        lanelines = outs[:, 4955:5483]
        lane_dict = {}
        oll, ll, rll, orl = np.split(lanelines, 4, axis=1)
        lane_dict["oll"] = oll.reshape(2, 33, 2)
        lane_dict["ll"] = ll.reshape(2, 33, 2)
        lane_dict["rll"] = rll.reshape(2, 33, 2)
        lane_dict["orl"] = orl.reshape(2, 33, 2)

        # Show raw camera image
        frame = cv2.resize(bgr_frames[i], (640, 420))
        cv2.imshow("modeld", frame)

        # Clean plot for next frame
        plt.clf()
        plt.title("lanes and path")

        plt.plot(lane_dict["ll"][LANE_MEAN, :, LANE_Y], X_IDXS, "b-", linewidth=1)
        plt.plot(lane_dict["rll"][LANE_MEAN, :, LANE_Y], X_IDXS, "r-", linewidth=1)
        plt.plot(lane_dict["oll"][LANE_MEAN, :, LANE_Y], X_IDXS, "m-", linewidth=1)
        plt.plot(lane_dict["orl"][LANE_MEAN, :, LANE_Y], X_IDXS, "k-", linewidth=1)
        plt.plot(best_path[PLAN_MEAN, :, PLAN_Y], best_path[PLAN_MEAN, :, PLAN_X], "g-", linewidth=1)

        plt.gca().invert_xaxis()
        plt.pause(0.001)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    plt.show()
