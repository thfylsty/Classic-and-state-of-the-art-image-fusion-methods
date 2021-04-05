import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import h5py

f = h5py.File('Dataset2.h5', 'r')
# for key in f.keys():
#   print(f[key].name)
a = f['data'][:]
sources = np.transpose(a, (0, 3, 2, 1))

vis = sources[100, :, :, 0]
ir = sources[100, :, :, 1]

ir_ds = scipy.ndimage.zoom(ir, 0.25)
ir_ds_us = scipy.ndimage.zoom(ir_ds, 4, order = 3)

fig = plt.figure()
V = fig.add_subplot(221)
I = fig.add_subplot(222)
I_ds = fig.add_subplot(223)
I_ds_us = fig.add_subplot(224)

V.imshow(vis, cmap = 'gray')
I.imshow(ir, cmap = 'gray')
I_ds.imshow(ir_ds, cmap = 'gray')
I_ds_us.imshow(ir_ds_us, cmap = 'gray')
plt.show()
# print
# 'Resampled by a factor of 2 with nearest interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 0)
#
# print
# 'Resampled by a factor of 2 with bilinear interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 1)
#
# print
# 'Resampled by a factor of 2 with cubic interpolation:'
# print
# scipy.ndimage.zoom(x, 2, order = 3)
#
# print
# 'Downsampled by a factor of 0.5 with default interpolation:'
# print(scipy.ndimage.zoom(x, 0.5))
