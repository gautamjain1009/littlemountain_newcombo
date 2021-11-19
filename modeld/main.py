# !/usr/bin/env python3
import onnxruntime
import sys
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from common.transformations.model import medmodel_intrinsics
from common.transformations.camera import transform_img, transform_img_cool, eon_intrinsics
import os

# from tools.lib.logreader import LogReader


TRAJECTORY_SIZE = 33
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


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


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

    pbar = tqdm(total=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    while cap.isOpened():
        # update tqdm
        pbar.update(1)

        ret, frame = cap.read()
        if not ret: break

        frames.append(frame)

    return frames


def load_calibration(segment_path):
    logs_file = os.path.join(segment_path, 'rlog.bz2')
    lr = LogReader(logs_file)
    liveCalibration = [m.liveCalibration for m in lr if m.which() == 'liveCalibration'] # probably not 1200, but 240
    return liveCalibration


def bgr_to_yuv(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YUV_I420)
    img_yuv = img_yuv.reshape((874*3//2, 1164))
    return img_yuv


def transform_frames_cool(frames, rpy):
    imgs = np.zeros((len(frames), 384, 512), dtype=np.uint8)
    for i, img in tqdm(enumerate(frames)):
        imgs[i] = transform_img_cool(img, rpy)
    return np.array(imgs)


def transform_frames(frames):
    imgs_med_model = np.zeros((len(frames), 384, 512), dtype=np.uint8)
    for i, img in tqdm(enumerate(frames)):
        imgs_med_model[i] = transform_img(img, from_intr=eon_intrinsics, to_intr=medmodel_intrinsics, yuv=True,
                                        output_size=(512, 256))
    return np.array(imgs_med_model)


def get_initial_inputs():
    recurrent_state = np.zeros((1, 512), dtype=np.float32)
    desire = np.zeros((1, 8), dtype=np.float32)
    drive_convention = np.zeros((1, 2), dtype=np.float32)
    drive_convention[0, 1] = 1.0 # right-hand-side

    return recurrent_state, desire, drive_convention


# adopted from https://github.com/commaai/openpilot/blob/v0.8.9/selfdrive/modeld/models/driving.cc#L240-L274
def parse_plan(data, columns=15, column_offset=0):
    x_arr = [0] * TRAJECTORY_SIZE
    y_arr = [0] * TRAJECTORY_SIZE
    z_arr = [0] * TRAJECTORY_SIZE
    x_std_arr = [0] * TRAJECTORY_SIZE
    y_std_arr = [0] * TRAJECTORY_SIZE
    z_std_arr = [0] * TRAJECTORY_SIZE

    for i in range(TRAJECTORY_SIZE):
        x_arr[i] = data[i*columns + 0 + column_offset]
        x_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 0 + column_offset]

        y_arr[i] = data[i*columns + 1 + column_offset]
        y_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 1 + column_offset]

        z_arr[i] = data[i*columns + 2 + column_offset]
        z_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 2 + column_offset]

    return x_arr, y_arr, z_arr, x_std_arr, y_std_arr, z_std_arr


# adopted from https://github.com/commaai/openpilot/blob/v0.8.9/selfdrive/modeld/models/driving.cc#L240-L274
def parse_laneline(data, columns=2, column_offset=-1):
    y_arr = [0] * TRAJECTORY_SIZE
    z_arr = [0] * TRAJECTORY_SIZE
    y_std_arr = [0] * TRAJECTORY_SIZE
    z_std_arr = [0] * TRAJECTORY_SIZE

    for i in range(TRAJECTORY_SIZE):
        y_arr[i] = data[i*columns + 1 + column_offset]
        y_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 1 + column_offset]

        z_arr[i] = data[i*columns + 2 + column_offset]
        z_std_arr[i] = data[columns*(TRAJECTORY_SIZE + i) + 2 + column_offset]

    return y_arr, z_arr, y_std_arr, z_std_arr


def forward_model(model, input_imgs, recurrent_state, desire, drive_convention):
    inputs = {'input_imgs': input_imgs,
              'initial_state': recurrent_state,
              'desire': desire,
              'traffic_convention': drive_convention
              }

    outs = model.run(None, inputs)[0]
    recurrent_state = outs[:, 5960:]

    assert recurrent_state.shape == (1, 512)

    return outs, recurrent_state


if __name__ == "__main__":
    # segment = '/data/realdata/aba20ae4/06b5d227e8d0f0a632c4d0a7bd9e8f6e305ed1910848cd5597eecdfe8e6d0023/1b9935697c8f4671ca355e2acd851d99e249c50c9948256972c4b6b4b045fc8f/2021-09-14--09-19-21/20'
    segment = sys.argv[1]

    input_video = os.path.join(segment, 'fcamera.hevc')
    # calibrations = load_calibration(segment)

    bgr_frames = load_frames(input_video)
    yuv_frames = [bgr_to_yuv(frame) for frame in bgr_frames]
    transformed_frames = transform_frames(yuv_frames)
    frame_tensors = frames_to_tensor(transformed_frames).astype(np.float32) # NOTE: should NOT be normalized

    model = load_model()
    plt.figure(figsize=(3, 10))

    recurrent_state, desire, drive_convention = get_initial_inputs()

    for i in tqdm(range(len(frame_tensors) - 1)):
        
        stacked_frames = np.vstack(frame_tensors[i:i+2])[None]
        assert stacked_frames.shape == (1, 12, 128, 256)

        outs, recurrent_state = forward_model(model, stacked_frames, recurrent_state, desire, drive_convention)

        """
        INDEXING OF THE FOLLOWING OUTPUT VECTOR OF SIZE (1,6472)
        As already mentioned here:-- https://github.com/commaai/openpilot/tree/master/models

        """
        # PLAN
        path_plans = outs[:, :4955]

        paths = np.array(np.split(path_plans, 5, axis=1))
        paths = paths.squeeze() # (5, 991)

        best_idx = np.argmax(paths[:, -1], axis=0)
        best_path = paths[best_idx, :-1].reshape(2, 33, 15)

        # LANELINES
        lanelines = outs[:, 4955:5483]
        lane_dict = {}
        oll, ll, rll, orl = np.split(lanelines, 4, axis=1)

        # using OP code
        oll_y, oll_z, oll_y_std, oll_z_std = parse_laneline(oll[0])
        ll_y, ll_z, ll_y_std, ll_z_std = parse_laneline(ll[0])
        rll_y, rll_z, rll_y_std, rll_z_std = parse_laneline(rll[0])
        orl_y, orl_z, orl_y_std, orl_z_std = parse_laneline(orl[0])

        # our custom interpretation
        lane_dict["oll"] = oll.reshape(2, 33, 2)
        lane_dict["ll"] = ll.reshape(2, 33, 2)
        lane_dict["rll"] = rll.reshape(2, 33, 2)
        lane_dict["orl"] = orl.reshape(2, 33, 2)

        # check our lanelines are correct (same as extracted with OP's `parse_laneline`)
        assert np.all(oll_y == lane_dict["oll"][LANE_MEAN, :, LANE_Y])
        assert np.all(ll_y == lane_dict["ll"][LANE_MEAN, :, LANE_Y])
        assert np.all(rll_y == lane_dict["rll"][LANE_MEAN, :, LANE_Y])
        assert np.all(orl_y == lane_dict["orl"][LANE_MEAN, :, LANE_Y])


        # LANELINE PROBS
        lanelines_probs = outs[:, 5483:5483+8]
        lanelines_probs = lanelines_probs.reshape(4, 2)
        lanelines_probs = sigmoid(lanelines_probs)
        print('lprobs (deprecated?):', lanelines_probs[:, 0])
        print('lprobs:', lanelines_probs[:, 1])

        oll_prob, ll_prob, rll_prob, orl_prob = lanelines_probs[:, 1].tolist()

        # Show raw camera image
        frame = cv2.resize(bgr_frames[i], (640, 420))
        cv2.imshow("modeld", frame)

        # Clean plot for next frame
        plt.clf()
        plt.title("lanes and path")
        plt.xlim(-10, 10)
        plt.ylim(-1, 196)



        # Plot lane lines FIXME: these are bogus, what's up?
        #
        # (almost) correct lanelines here: (graphically but not semantically)
        # https://github.com/MTammvee/openpilot-supercombo-model
        plt.plot(ll_y, X_IDXS, "b-", linewidth=1, alpha=ll_prob)
        plt.plot(rll_y, X_IDXS, "r-", linewidth=1, alpha=rll_prob)
        plt.plot(oll_y, X_IDXS, "m-", linewidth=1, alpha=oll_prob)
        plt.plot(orl_y, X_IDXS, "k-", linewidth=1, alpha=orl_prob)

        # best path plan
        plt.plot(best_path[PLAN_MEAN, :, PLAN_Y], best_path[PLAN_MEAN, :, PLAN_X], "g-", linewidth=1)

        # plt.gca().invert_xaxis()
        plt.pause(0.001)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    plt.show()
