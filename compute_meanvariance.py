### Copyright (C) 2017 NVIDIA Corporation. All rights reserved.
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import re
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from glob import glob
from pathlib import Path
from time import time

import json
from tqdm import tqdm
from PIL import Image
import torch
from torch import nn
import shutil
import video_utils
import image_transforms
import argparse
from data.multi_frame_dataset import MultiFrameDataset

class MeanVarOptions(TestOptions):
    def __init__(self):
        TestOptions.__init__(self)
        self.parser.add_argument('--root-dir', help='dir containing the two classes folders', dest="root_dir")
        self.parser.add_argument('--gpu', type=bool, default=False, help='Train on GPU')

opt = MeanVarOptions().parse(save=False)
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# additional enforced options for video
opt.video_mode = True
opt.label_nc = 0
opt.no_instance = True
opt.resize_or_crop = "none"

model = create_model(opt)

# Not real code TODO change with opt as MultiFrameDataset wants .initialize()...
print('Processing has_target folder')
has_tgt = MultiFrameDataset()
opt.dataroot = str(Path(opt.root_dir) / "has_target")
has_tgt.initialize(opt)

all_times = []
all_differences = []
path_differences = {}
with torch.no_grad():
    for i, data in enumerate(tqdm(has_tgt)):
        left_frame = Image.open(data['left_path'])
        real_right_frame = Image.open(data['right_path'])

        left_frame = video_utils.im2tensor(left_frame)
        real_right_frame = video_utils.im2tensor(real_right_frame)

        if opt.gpu:
            left_frame = left_frame.to('cuda')
            real_right_frame = real_right_frame.to('cuda')

        t0 = time()
        generated_right_frame = video_utils.next_frame_prediction(model, left_frame)
        t1 = time()
        all_times.append(t1 - t0)
        loss = nn.MSELoss()
        cur_loss = float(loss(generated_right_frame, real_right_frame))

        fname = Path(data['left_path']).parent.name
        if fname not in path_differences:
            path_differences[fname] = []
        path_differences[fname].append(cur_loss)
        all_differences.append(cur_loss)

print('Saving has_tgt_differences')
with open(Path(opt.dataroot) / 'has_tgt_differences.json', 'w') as fout:
    json.dump(path_differences, fout)

print('Processing normal folder')
no_tgt = MultiFrameDataset()
opt.dataroot = str(Path(opt.root_dir) / "normal")
no_tgt.initialize(opt)

path_differences = {}
with torch.no_grad():
    for i, data in enumerate(no_tgt):
        left_frame = Image.open(data['left_path'])
        real_right_frame = Image.open(data['right_path'])

        left_frame = video_utils.im2tensor(left_frame)
        real_right_frame = video_utils.im2tensor(real_right_frame)

        if opt.gpu:
            left_frame = left_frame.to('cuda')
            real_right_frame = real_right_frame.to('cuda')

        t0 = time()
        generated_right_frame = video_utils.next_frame_prediction(model, left_frame)
        t1 = time()
        all_times.append(t1 - t0)
        loss = nn.MSELoss()
        cur_loss = float(loss(generated_right_frame, real_right_frame))

        fname = Path(data['left_path']).parent.name
        if fname not in path_differences:
            path_differences[fname] = []
        path_differences[fname].append(cur_loss)
        all_differences.append(cur_loss)

avg_time = sum(all_times) / len(all_times)
print(f'Average inference time: {avg_time}')
# avgt = {'avg_time': avg_time}
# with open(Path(opt.dataroot) / 'avg_time.json', 'w') as f:
#     json.dump(avgt, f)

print('Saving no_tgt_differences')
# with open(Path(opt.dataroot) / 'no_tgt_differences.json', 'w') as fout:
#     json.dump(path_differences, fout)

# print('Saving all_differences')
# with open(Path(opt.dataroot) / 'all_differences.txt', 'w') as f:
#     for item in all_differences:
#         f.write(f"{item}\n")

# Now all_differences[] contains all the deltas between generated frames and real frames
print('Computing mean and variance')
mean = torch.mean(torch.FloatTensor(all_differences))
std = torch.std(torch.FloatTensor(all_differences))

print(f'MEAN: {mean.item()}')
print(f'STD: {std.item()}')

ms = {'mean': mean, 'std':std}
with open(Path(opt.dataroot) / 'mean_std.json', 'w') as f:
    json.dump(ms, f)
