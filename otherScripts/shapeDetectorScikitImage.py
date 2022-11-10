# https://peerj.com/articles/453/ 
# https://scikit-image.org/
# https://peerj.com/articles/453/#fig-1
# pip install -U scikit-image

#%%
#-----------------------------------------------------------------------#
from skimage import (data, io, )
from skimage import filters 
from skimage import feature 
from PIL import ImageFilter

image = data.coins()  # or any NumPy array!
edges = filters.sobel(image)
io.imshow(edges)
#-----------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt

# Load a small section of the image.
image = data.coins()[0:95, 70:370]

fig, axes = plt.subplots(ncols=2, nrows=3,
                         figsize=(8, 4))
ax0, ax1, ax2, ax3, ax4, ax5  = axes.flat
ax0.imshow(image, cmap=plt.cm.gray)
ax0.set_title('Original', fontsize=24)
ax0.axis('off')

#-----------------------------------------------------------------------#
# Histogram.
values, bins = np.histogram(image,
                            bins=np.arange(256))

ax1.plot(bins[:-1], values, lw=2, c='k')
ax1.set_xlim(xmax=256)
ax1.set_yticks([0, 400])
ax1.set_aspect(.2)
ax1.set_title('Histogram', fontsize=24)

#-----------------------------------------------------------------------#
# Apply threshold.
from skimage.filters import threshold_local

bw = threshold_local(image, 95, offset=-15)

ax2.imshow(bw, cmap=plt.cm.gray)
ax2.set_title('Adaptive threshold', fontsize=24)
ax2.axis('off')

#-----------------------------------------------------------------------#
# Find maxima.
from skimage.feature import peak_local_max

coordinates = peak_local_max(image, min_distance=20)

ax3.imshow(image, cmap=plt.cm.gray)
ax3.autoscale(False)
ax3.plot(coordinates[:, 1],
         coordinates[:, 0], c='red')
ax3.set_title('Peak local maxima', fontsize=24)
ax3.axis('off')

#-----------------------------------------------------------------------#
# Detect edges.

edges = feature.canny(image, sigma=3,
                     low_threshold=10,
                     high_threshold=80)

ax4.imshow(edges, cmap=plt.cm.gray)
ax4.set_title('Edges', fontsize=24)
ax4.axis('off')

#-----------------------------------------------------------------------#
# Label image regions.
from skimage.measure import regionprops
import matplotlib.patches as mpatches
from skimage.morphology import label

label_image = label(edges)

ax5.imshow(image, cmap=plt.cm.gray)
ax5.set_title('Labeled items', fontsize=24)
ax5.axis('off')

for region in regionprops(label_image):
    # Draw rectangle around segmented coins.
    minr, minc, maxr, maxc = region.bboxfeature
    rect = mpatches.Rectangle((minc, minr),
                              maxc - minc,
                              maxr - minr,
                              fill=False,
                              edgecolor='red',
                              linewidth=2)
    ax5.add_patch(rect)

plt.tight_layout()
plt.show()

#%% https://scikit-image.org/docs/stable/auto_examples/edges/plot_canny.html


print('STOP MDR SALE PUTE')
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from skimage.util import random_noise
from skimage import feature


# Generate noisy image of a square
image = np.zeros((128, 128), dtype=float)
image[32:-32, 32:-32] = 1

image = ndi.rotate(image, 15, mode='constant')
image = ndi.gaussian_filter(image, 4)
image = random_noise(image, mode='speckle', mean=0.1)

# Compute the Canny filter for two values of sigma
edges1 = feature.canny(image)
edges2 = feature.canny(image, sigma=3)

# display results
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(8, 3))

ax[0].imshow(image, cmap='gray')
ax[0].set_title('noisy image', fontsize=20)

ax[1].imshow(edges1, cmap='gray')
ax[1].set_title(r'Canny filter, $\sigma=1$', fontsize=20)

ax[2].imshow(edges2, cmap='gray')
ax[2].set_title(r'Canny filter, $\sigma=3$', fontsize=20)

for a in ax:
    a.axis('off')

fig.tight_layout()
plt.show()
#%% 
A = {
        '01': (242, 233), '02': (274, 233), '03': (305, 233), '04': (336, 233), '05': (367, 233),
        '11': (242, 262), '12': (274, 262), '13': (305, 262), '14': (336, 262), '15': (367, 262),
        '21': (242, 294), '22': (274, 294), '23': (305, 294), '24': (336, 294), '25': (367, 294),
        '31': (242, 324), '32': (274, 324), '33': (305, 324), '34': (336, 324), '35': (367, 324),
        '41': (242, 354), '42': (274, 354), '43': (305, 354), '44': (336, 354), '45': (367, 354),
        '51': (242, 383), '52': (274, 383), '53': (305, 383), '54': (336, 383), '55': (367, 383),
        '61': (242, 412), '62': (274, 412), '63': (305, 412), '64': (336, 412), '65': (367, 412),
    }

from utils import dirpath, locateFileExt, locateFileNameExt
import numpy as np
import os

basepath_ = os.path.join(dirpath(),'canaux')
basepaths = locateFileNameExt(path=basepath_, containInName="stackImagePower5.0um", ext=".npy")
basepaths.sort() 
print([basepath.split('\\')[-1].split('.')[1] for basepath in basepaths])

stackImages = np.load(basepaths[0])
Aimages = {f"{list(A.keys())[i]}":image for i, image in enumerate(stackImages)}
img = Aimages['43']
img_ = img.reshape(img.shape[0],int(np.sqrt(img.shape[1])),int(np.sqrt(img.shape[1])))

import matplotlib.pyplot as plt 

#%
from utils import masking
step, coef = 1000, 3
minValue, maxValue = np.min(img_[0,:,:]) + coef*step, np.max(img_[0,:,:]) - coef*step*0
img = masking(img_[0,:,:], minValue, maxValue)
plt.imshow(img, cmap='gray')

#%%
# Background cancellation 
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rolling_ball.html#sphx-glr-auto-examples-segmentation-plot-rolling-ball-py

# %%
