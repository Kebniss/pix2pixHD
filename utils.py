import os
import ntpath
import shutil


def get_filename_extension(path):
    basename = ntpath.basename(path).split('.')
    return basename[0], basename[1]


def rm_mkdir(folder_path):
    # if folder_path exists deletes it and creates it new
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    else:
        shutil.rmtree(folder_path)
        os.mkdir(folder_path)
