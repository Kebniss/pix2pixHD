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
for f in tqdm(frames):
    current_frame = video_utils.im2tensor(Image.open(f))
    next_frame = video_utils.next_frame_prediction(model, current_frame)

    video_utils.save_tensor(
        next_frame,
        frame_dir + "/frame-%s.jpg" % str(frame_index).zfill(5),
    )
    frame_index += 1
    prev = current_frame

duration_s = frame_index / opt.fps
video_id = "epoch-%s_%s_%.1f-s_%.1f-fps" % (
    str(opt.which_epoch),
    opt.name,
    duration_s,
    opt.fps
)

video_path = output_dir + "/" + video_id + ".mp4"
while os.path.isfile(video_path):
    video_path = video_path[:-4] + "-.mp4"

video_utils.video_from_frame_directory(
    frame_dir,
    video_path,
    framerate=opt.fps,
    crop_to_720p=True,
    reverse=False
)

print("video ready:\n%s" % video_path)
