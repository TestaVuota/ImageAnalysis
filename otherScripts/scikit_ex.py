#%%
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:39:54 2022

@author: nicol

scikit_ex.py
"""
import numpy as np
from scipy import signal
from scipy import misc
rng = np.random.default_rng()
face = misc.face(gray=True) - misc.face(gray=True).mean()
template = np.copy(face[300:365, 670:750])  # right eye
template -= template.mean()
face = face + rng.standard_normal(face.shape) * 50  # add noise
corr = signal.correlate2d(face, template, boundary='symm', mode='same')
y, x = np.unravel_index(np.argmax(corr), corr.shape)  # find the match
import matplotlib.pyplot as plt
fig, (ax_orig, ax_template, ax_corr) = plt.subplots(3, 1,
                                                    figsize=(6, 15))
ax_orig.imshow(face, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()
ax_template.imshow(template, cmap='gray')
ax_template.set_title('Template')
ax_template.set_axis_off()
ax_corr.imshow(corr, cmap='gray')
ax_corr.set_title('Cross-correlation')
ax_corr.set_axis_off()
ax_orig.plot(x, y, 'ro')
fig.show()

#%%

import numpy as np
# function to be applied to the array
def add(num):
    return num+10
# creating  numpy array
arr = np.array([1, 2, 3, 4, 5])
# printing the original array
print(" The original array : " , arr)
# Apply add() function to array. 
arr = np.array(list(map(add, arr)))
# printing the array after applying function
print(" The array after applying function : " , arr)

#%%