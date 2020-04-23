from PIL import Image as im
import numpy
import cv2
import matplotlib.pyplot as plt

image = im.open('dog.jpg')
ycbcr = image.convert('YCbCr')

# From https://stackoverflow.com/questions/39939595/save-different-channels-of-ycbcr-as-seperate-images-python
def ycbcr_grayscale_more_complext_way():
  # output of ycbcr.getbands() put in order
  Y = 0
  Cb = 1
  Cr = 2

  YCbCr=list(ycbcr.getdata()) # flat list of tuples
  # reshape
  imYCbCr = numpy.reshape(YCbCr, (image.size[1], image.size[0], 3))
  # Convert 32-bit elements to 8-bit
  imYCbCr = imYCbCr.astype(numpy.uint8)

  # now, display the 3 channels
  im.fromarray(imYCbCr[:,:,Y], "L").show()
  im.fromarray(imYCbCr[:,:,Cb], "L").show()
  im.fromarray(imYCbCr[:,:,Cr], "L").show()

def test_lena():
  # test = numpy.array(im.open('lena.png').convert('YCbCr'))
  # test[:,:,0] *= 0
  # test[:,:,1] *= 0
  # test = im.fromarray(test, mode='YCbCr')
  # test.show()
  (r, g, b) = im.open('lena.png').split()
  r.show()
  g.show()
  b.show()

# for YCbCr grayscale
# From https://stackoverflow.com/questions/39939595/save-different-channels-of-ycbcr-as-seperate-images-python
def ycbcr_grayscale():

  (y, cb, cr) = ycbcr.split()
  y.save('y.png')
  cb.save('cb.png')
  cr.save('cr.png')


# FOR YCbCr COLORED
def ycbcr_colored():
  ycbcrdata = ycbcr.getdata()

  # need to put mid value here instead of 0, got hint from
  # http://rodrigoberriel.com/2014/11/opencv-color-spaces-splitting-channels/
  # https://stackoverflow.com/questions/33133372/splitting-ycrcb-image-to-its-intensity-channels
  # https://stackoverflow.com/questions/28638848/displaying-y-cb-and-cr-components-in-matlab/28639973#28639973
  # https://stackoverflow.com/a/43988642/4162778
  # https://stackoverflow.com/questions/30253719/opencv-display-colored-cb-cr-channels
  yColor = [(d[0], 127, 127) for d in ycbcrdata]
  cbColor = [(127, d[1], 127) for d in ycbcrdata]
  crColor = [(127, 127, d[2]) for d in ycbcrdata]

  ycbcr.putdata(yColor)
  ycbcr.save('yColor.jpg')
  ycbcr.putdata(cbColor)
  ycbcr.save('cbColor.jpg')
  ycbcr.putdata(crColor)
  ycbcr.save('crColor.jpg')

def rgb_greyscale():
  (r, g, b) = image.split()
  r.save('rGrey.png')
  g.save('gGrey.png')
  b.save('bGrey.png')

# FOR RGB COLORED
# From https://stackoverflow.com/questions/51325224/python-pil-image-split-to-rgb
def rgb_colored():
  data = image.getdata()

  # Suppress specific bands (e.g. (255, 120, 65) -> (0, 120, 0) for g)
  r = [(d[0], 0, 0) for d in data]
  g = [(0, d[1], 0) for d in data]
  b = [(0, 0, d[2]) for d in data]

  image.putdata(r)
  image.save('r.png')
  image.putdata(g)
  image.save('g.png')
  image.putdata(b)
  image.save('b.png')

def rgb2ycbcr(im):
  xform = numpy.array([[.299, .587, .114], [-.1687, -.3313, .5], [.5, -.4187, -.0813]])
  ycbcr = im.dot(xform.T)
  ycbcr[:,:,[1,2]] += 128

  result = numpy.uint8(ycbcr)

  # splitting following http://corochann.com/basic-image-processing-tutorial-1220.html
  rOnly = result.copy()
  rOnly[:, :, 1:3]

  gOnly = result.copy()
  gOnly[:, :, [0, 2]] = 0

  bOnly = result.copy()
  bOnly[:, :, 0:2] = 0

  plt.imshow(rOnly)
  plt.show()

  return (result, rOnly, gOnly,bOnly)

# rgb2ycbcr(numpy.array(im.open('lena.png')))

# from https://polprog.net/blog/ycc/
def ycbcr2(imag):
  COEF_R = 0.2126
  COEF_G = 0.7152
  COEF_B = 0.0722

  red = imag[:, :, 0]
  grn = imag[:, :, 1]
  blu = imag[:, :, 2]
      
  luma = imag[:, :, 0] * COEF_R + imag[:, :, 1] * COEF_G + imag[:, :, 2] * COEF_B

  pb = luma - blu
  pr = luma - red

  plt.imshow(pr)
  plt.show()

# ycbcr2(numpy.array(im.open('lena.png')))

def use_cv2_yuv():
  def make_lut_u():
    return numpy.array([[[i,255-i,0] for i in range(256)]],dtype=numpy.uint8)
  def make_lut_v():
      return numpy.array([[[0,255-i,i] for i in range(256)]],dtype=numpy.uint8)

  img = cv2.imread('lena.png')

  img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
  y, u, v = cv2.split(img_yuv)

  lut_u, lut_v = make_lut_u(), make_lut_v()

  # Convert back to BGR so we can apply the LUT and stack the images
  y = cv2.cvtColor(y, cv2.COLOR_GRAY2BGR)
  u = cv2.cvtColor(u, cv2.COLOR_GRAY2BGR)
  v = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

  u_mapped = cv2.LUT(u, lut_u)
  v_mapped = cv2.LUT(v, lut_v)

  result = numpy.vstack([img, y, u_mapped, v_mapped])

  cv2.imwrite('lena-combo.png', result)
