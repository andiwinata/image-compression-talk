import numpy as np
import matplotlib.pyplot as plt
import scipy
import cv2

from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
from scipy import misc

# https://stackoverflow.com/questions/9777783/suppress-scientific-notation-in-numpy-when-creating-array-from-nested-list
# prevent numpy exponential 
# notation on print, default False
np.set_printoptions(suppress=True)

# From https://en.wikipedia.org/wiki/JPEG#JPEG_codec_example
wiki_raw = np.array([
  [52, 55, 61, 66, 70, 61, 64, 73],
  [63, 59, 55, 90, 109, 85, 69, 72],
  [62, 59, 68, 113, 144, 104, 66, 73],
  [63, 58, 71, 122, 154, 106, 70, 69],
  [67, 61, 68, 104, 126, 88, 68, 70],
  [79, 65, 60, 70, 77, 68, 58, 75],
  [85, 71, 64, 59, 55, 61, 65, 83],
  [87, 79, 69, 68, 65, 76, 78, 94]
])

jpg_low_color = np.array([
  [255, 255, 255, 255,  255, 255, 255, 255],
  [255, 255, 255, 255,  255, 255, 255, 0  ],
  [255, 255, 255, 255,  255, 255, 0  , 0  ],
  [255, 255, 255, 255,  0  , 0  , 0  , 255],
  [255, 255, 255, 0  ,  0  , 255, 255, 255],
  [255, 0  , 0  , 0  ,  255, 255, 255, 255],
  [0  , 255, 255, 255,  255, 255, 255, 255],
  [255, 255, 255, 255,  255, 255, 255, 255],
])

def normalizeImg(img):
  # This normalization is wrong
  # return np.array(img) / 255

  # normalized this way following wiki JPEG
  # and https://stackoverflow.com/questions/31949210/assertion-failed-type-cv-32fc1-type-cv-64fc1-in-dct
  return (np.array(img) - 128) / 128

wiki_raw_normalized = normalizeImg(wiki_raw)
jpg_low_color_normalized = normalizeImg(jpg_low_color)

q50_table = np.array([
  [16, 11, 10, 16, 24, 40, 51, 61],
  [12, 12, 14, 19, 26, 58, 60, 55],
  [14, 13, 16, 24, 40, 57, 69, 56],
  [14, 17, 22, 29, 51, 87, 80, 62],
  [18, 22, 37, 56, 68, 109, 103, 77],
  [24, 35, 55, 64, 81, 104, 113, 92],
  [49, 64, 78, 87, 103, 121, 120, 101],
  [72, 92, 95, 98, 112, 100, 103, 99]
])

# from https://inst.eecs.berkeley.edu/~ee123/sp16/Sections/JPEG_DCT_Demo.html
def convert_dct():
  def dct2(a):
    return scipy.fft.dct( scipy.fft.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

  img = np.array(wiki_raw_normalized)
  # same result as convert_dct_cv2
  print(np.round(dct2(img) * 128, decimals=2))

# convert_dct()

def convert_dct_cv2(img):
  print('Start DCT')
  print(img)
  print('-----------')
  DCT = cv2.dct(img)
  result = np.round(DCT * 128, decimals=2)
  print('Coefficient')
  print(result)
  print('-----------')
  return result

def quantization(coefficient):
  result = np.round(coefficient / q50_table)
  print('After quantization')
  print(result)
  print('-----------')
  return result

quantization(convert_dct_cv2(wiki_raw_normalized))
quantization(convert_dct_cv2(jpg_low_color_normalized))
