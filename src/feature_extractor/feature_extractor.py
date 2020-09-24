if __name__ == "__main__":  
  import sys
  sys.path.append('/home/jonas/Documents/Repos/Visual-Odom-Pipeline')
  sys.path.append('/home/jonas/Documents/Repos/Visual-Odom-Pipeline/src')
  from loader import ParkingLoader

import numpy as np 
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt

def extract_features(img):
  # Initiate STAR detector
  orb = cv2.ORB_create()
  # find the keypoints with ORB
  kp = orb.detect(img,None)
  # compute the descriptors with ORB
  kp, des = orb.compute(img, kp)
  # draw only keypoints location,not size and orientation
  img2 = cv2.drawKeypoints(img,kp,img, color=(0,255,0), flags=0)
  return kp,des, img2


def test():
  
  pl = ParkingLoader(root='/home/jonas/Documents/Datasets/parking')
  img1 = pl.getImage(0)
  img2 = pl.getImage(10)
  kp1,des1, img_kp1 = extract_features(img1)
  kp2,des2, img_kp2 = extract_features(img2)

  f, axarr = plt.subplots(2,2)
  axarr[0,0].imshow(img_kp1)
  axarr[0,1].imshow(img_kp2)
  axarr[1,0].imshow(img1)
  axarr[1,1].imshow(img2)
  plt.show()

  # create BFMatcher object
  bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

  # Match descriptors.
  matches = bf.match(des1,des2)

  # Sort them in the order of their distance.
  matches = sorted(matches, key = lambda x:x.distance)

  # Draw first 10 matches.

  img_matches = cv2.drawMatches(img1,kp1,img2,kp2,matches[:10], img1)

  plt.imshow(img_matches)
  plt.show()
  
if __name__ == "__main__":
  test()		