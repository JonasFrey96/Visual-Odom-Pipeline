"""Generate video from an image directory"""
import cv2
import os
from glob import glob
from tqdm import tqdm

# Input Arguments
image_dir = '/media/carter/Samsung_T5/vo_data/logged_runs/bundle_adjusted/custom'
frame_rate = 5

# Paths
image_paths = sorted(glob(os.path.join(image_dir, '*')))
_, stem = os.path.split(image_dir)
output_path = os.path.join(image_dir, '..', stem+'.avi')

# Initialize Output Video
im_test = cv2.imread(image_paths[0])
frame_height, frame_width = im_test.shape[:2]
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc('M','J','P','G'), frame_rate, (frame_width, frame_height))

# Write frames
for im_p in tqdm(image_paths):
    im = cv2.imread(im_p)
    out.write(im)