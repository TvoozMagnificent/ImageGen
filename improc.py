
factor = 7
threshold = 100

import imageio as iio
import numpy

img = iio.imread("im15.png")[::factor, ::factor, 0]
img = numpy.where(img < threshold, 0, 255)
iio.imwrite("new15.png", img)
