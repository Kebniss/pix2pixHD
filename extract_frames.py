import os
import cv2
import argparse
from utils import *
from tqdm import tqdm
from glob import glob
from pathlib import Path


def _extract_frames(video_path, parent, start=0, sampling_f=1):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    count = 0
    while success:
        if not count % sampling_f == 0:
            # sampling
            continue
        cv2.imwrite(''.join([dest_folder, f"/{count + start}.jpg"]), image)
        success, image = vidcap.read()  # read next
        count += 1
    print(f'Successfully saved {count} frames to {dest_folder}')
    return count + start


parser = argparse.ArgumentParser(
    description='build a "frame dataset" from a given video')
parser.add_argument('-input', dest="input", required=True,
    help='''Path to a single video or a folder. If path to folder the algorithm
         will extract frames from all .mov files and save them under separate
         folders under dest_folder. The frames from each video will be saved
         under a folder with its name.''')
parser.add_argument('--dest-folder', dest="dest_folder", default='./dataset/',
    help='''Path where to store frames. NB all files in this folder will be
         removed before adding the new frames''')
parser.add_argument('--same-folder', dest="same_folder", default=False,
    help='''Set it to True if you want to save the frames of all videos to the
    same folder in ascending order going from the first frame of the first video
    to the last frame of the last video. If True frames will be saved in
    dest_folder/frames.''')
parser.add_argument('--run-type', help='train or test', default='train')
parser.add_argument('-width', help='output width', default=640, type=int)
parser.add_argument('-height', help='output height', default=480, type=int)
args = parser.parse_args()

mkdir(args.dest_folder)

if (args.width % 32 != 0) or (args.height % 32 != 0):
    raise Exception("Please use width and height that are divisible by 32")

if os.path.isdir(args.input):
    inp = str(Path(args.input) / '*.mov')
    videos = [v for v in glob(inp)]
    if not videos:
        raise Exception(f'No .mov files in input directory {args.input}')
elif os.path.isfile(args.input):
    _, ext = get_filename_extension(args.input)
    if ext != '.mov':
        raise ValueError('Correct inputs: folder or path to avi file only')
    videos = [args.input]
else:
    raise ValueError('Correct inputs: folder or path to mov file only')

if args.same_folder:
    start = 0
    dest_folder = str(Path(args.dest_folder) / f'{args.run_type}_frames')
    mkdir(dest_folder)

for v in tqdm(videos):
    if not args.same_folder:
        start = 0
        name, _ = get_filename_extension(video_path)
        dest_folder = str(Path(args.dest_folder) / name)
        mkdir(dest_folder)

    start = _extract_frames(v, dest_folder, start)
