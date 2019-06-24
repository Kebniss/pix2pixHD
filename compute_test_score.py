### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import re
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from glob import glob
from pathlib import Path

from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import shutil
import video_utils
import image_transforms


fname = re.compile('(\d+).jpg')


def extract_name(path):
    return int(fname.search(path).group(1))


opt = TestOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

# loading initial frames from: ./datasets/NAME/test_frames
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

# this directory will contain the generated videos
output_dir = os.path.join(opt.checkpoints_dir, opt.name, 'output')
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)

# this directory will contain the frames to build the video
frame_dir = os.path.join(opt.checkpoints_dir, opt.name, 'frames')
if os.path.isdir(frame_dir):
    shutil.rmtree(frame_dir)
os.mkdir(frame_dir)

frame_index = 1

frames_path = opt.start_from
if os.path.isdir(frames_path):
    frames = [f for f in glob(str(Path(frames_path) / '*.jpg'))]
    frames = sorted(frames, key=extract_name)
else:
    raise ValueError('Please provide the path to a folder with frames.jpg')

model = create_model(opt)

frames_count = 1
next_frame = torch.Tensor()

# Not real code TODO change with opt as MultiFrameDataset wants .initialize()...
positives = MultiFrameDataset("datasets/insight_dataset/validation/has_anomaly")
negatives = MultiFrameDataset("datasets/insight_dataset/validation/norm")

# This one has MultiFrameDataset's init hardcoded inside. Change it to the right folders
# from there...
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()

dataset_size = len(data_loader)

differences = []

preds = []
with torch.no_grad():
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        
        left_frame = Image.open(data['left_path'][0])
        real_right_frame = Image.open(data['right_path'][0])

        left_frame = video_utils.im2tensor(left_frame)
        real_right_frame = video_utils.im2tensor(right_frame)

        if opt.gpu:
            left_frame = left_frame.to('cuda')
            real_right_frame = real_right_frame.to('cuda')

        generated_right_frame = video_utils.next_frame_prediction(model, left_frame)
        loss = nn.MSELoss()
        loss_score = float(loss(generated_right_frame, real_right_frame))
        y_pred = 'has_anomaly' if loss_score > mean + 2 * std else 'normal'
        y_true = ... # TODO read from folder eg datasets/insight_dataset/validation/**has_anomaly**
        preds.append((y_pred, y_true))


import json
with open(..., 'w') as fout:
    json.dump(preds, fout)


from sklearn.metrics import f1score

f1score(y_pred, y_true)
...