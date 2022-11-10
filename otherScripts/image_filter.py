# -*- coding: utf-8 -*-

"""
Created on Wed May  4 14:02:58 2022

@author: nicol
"""
#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator


pathimg = r'C:\Users\nicol\Documents\Python\lulu\PortII\750.tiff'
image = cv2.imread(pathimg)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# class filter():
# Create a custom kernel

# src : https://alyxion.github.io/Udacity_IntroToSelfDrivingCarsNd/8_1_Intro_Mini_Projects/41_Finding%20Edges%20and%20Custom%20Kernels.html
# 3x3 array for edge detection
sobel_y = np.array([
                    [ -1, -2, -1], 
                    [  0,  0,  0], 
                    [  1,  2,  1]
                ])

## TODO: Create and apply a Sobel x operator
sobel_x = np.array([
                    [ -1, 0, 1], 
                    [ -2, 0, 2], 
                    [ -1, 0, 1]
                ])

# scharr sobel filter 
scharr_sobel = sobel_x+sobel_y*1j

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_y = cv2.filter2D(gray, -1, sobel_y)

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

fig1, axes = plt.subplots(2, 2, figsize=(15, 20), constrained_layout=False)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

fig1.suptitle('Images filtered', fontsize=20)

ax1.set_title('sobel x')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im1 = ax1.imshow(filtered_image_x, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider1 = make_axes_locatable(ax1)
# Append axes to the right of ax1, with 20% width of ax1
cax1 = divider1.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar1 = plt.colorbar(im1, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax1
ax1.xaxis.set_visible(False)

ax2.set_title('sobel y')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im2 = ax2.imshow(filtered_image_y, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider2 = make_axes_locatable(ax2)
# Append axes to the right of ax2, with 10% width of ax2
cax2 = divider2.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar2 = plt.colorbar(im2, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax2
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)

blur_filter1= np.array([
                        [ 0.05, 0.2, 0.05], 
                        [  0.2, 0.2,  0.2], 
                        [ 0.05, 0.2, 0.05]
                    ])
blur_filter2 = blur_filter1*0.1

L = np.array([
    [  0,  0, -1,  0,  0],
    [  0, -1, -2, -1,  0],
    [ -1, -2,-16, -2, -1],
    [  0, -1, -2, -1,  0],
    [  0,  0, -1,  0,  0]
]) #best

edge_f = np.array([
                [-1,-1,-1],
                [-1,-8,-1],
                [-1,-1,-1]
            ])

def Normal_f(x,mu,sigma):
    G = np.exp((x-mu/sigma)**2*(-1/2))/(sigma*2*np.pi)
    return G

def Gab(c,sigma):
    m,n=(c,c)
    G = np.zeros((m,n))
    for a in range(m):
        for b in range(n):
            
            if c/2>int(c/2):
                G[a,b] = np.exp(((a-int((m)/2))**2+(b-int((n)/2))**2)*(-0.5)*(1/sigma)**2)#/sum(np.exp(((a-(m+1)/2)**2+(b-(n+1)/2)**2)*(-1/2)))
           
            else:
                pass
                G[a,b] = np.exp(((a-int((m+1)/2))**2+(b-int((n+1)/2))**2)*(-0.5)*(1/sigma)**2)#/sum(np.exp(((a-(m+1)/2)**2+(b-(n+1)/2)**2)*(-1/2)))
   
    return G/sum(sum(G))

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_blur = cv2.filter2D(gray, -1, blur_filter)

#------------------------------------------------------------------------------------
edge_detection = np.array([
                    [ 0.0, -1,  0], 
                    [  -1,  4, -1], 
                    [   0, -1,  0]
                ])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_edges = cv2.filter2D(filtered_image_blur, -1, edge_detection)

ax3.set_title('image_blur')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im3 = ax3.imshow(filtered_image_blur, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider3 = make_axes_locatable(ax3)
# Append axes to the right of ax3, with 30% width of ax3
cax3 = divider3.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax3
ax3.xaxis.set_visible(False)

ax4.set_title('image_edges')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax4)
im4 = ax4.imshow(filtered_image_edges, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider4 = make_axes_locatable(ax4)
# Append axes to the right of ax4, with 40% width of ax4
cax4 = divider4.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar4 = plt.colorbar(im4, cax=cax4, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax4
ax4.yaxis.set_visible(False)
ax4.xaxis.set_visible(False)


#%%
# plot sauvage !!!
hey = Gab(10,1.5) ;x = np.linspace(0,9,10); plt.plot(x,hey);plt.grid();plt.title('gaussian_shape')

#%%
# Heat map on gray image
from scipy.ndimage import gaussian_filter;import matplotlib.pyplot as plt
result = gaussian_filter(gray, sigma=0.5)
plt.imshow(result, cmap='hsv')
cbar = plt.colorbar()
cbar.solids.set_edgecolor("face")
plt.draw()
#%%
# Gradient Magnitude and Gradient Orientation
# src : https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/#pyis-cta-modal
from scipy import signal
from scipy import misc

ascent = misc.ascent()
acsent = gray
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

grad = signal.convolve2d(gray, scharr, boundary='symm', mode='same')
import matplotlib.pyplot as plt
fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3, figsize=(15, 15))

ax_orig.imshow(gray, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_mag.imshow(np.absolute(grad), cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
ax_ang.set_title('Gradient orientation')
ax_ang.set_axis_off()

fig.show()

#%%
# filtre fin edge detecter, on voit les microstructure
# src : https://www.geeksforgeeks.org/python-edge-detection-using-pillow/
from PIL import Image,ImageFilter
import cv2

im  = Image.fromarray(cv2.imread(pathimg)) 
res = im.filter(ImageFilter.CONTOUR)
res.show('hsv')
final = im.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                 -1, -1, -1, -1), 1, 0))

final.show()
# plt.contour(np.reshape(im.getdata(),im.size))
#%%
# # plot 3D qui ne fonctionne pas
# from mpl_toolkits.mplot3d import Axes3D
# xe,ye = gray.shape
# X,Y = np.meshgrid(np.linspace(0,xe-1,xe),np.linspace(0,ye-1,ye))
# Z=np.asarray(gray).T

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_figs(fig_num, elev, azim, Z):
#     fig = plt.figure(fig_num, figsize=(8, 8))
#     plt.clf()
#     ax = Axes3D(fig, elev=elev, azim=azim)

#     my_cmap= plt.get_cmap('cool')
#     surf = ax.plot_surface(
#         X, Y, Z,
#         alpha=0.3,
#         cmap = my_cmap,
#         edgecolor = 'none'
#     )
#     fig.colorbar(surf, ax = ax, shrink = 0.7 , aspect = 7)
#     plt.legend()

# # Generate the three different figures from different views
# elev = 43.5
# azim = -110
# plot_figs(1, elev, azim, 2)
# # %%
#%%
# Article à lire
# https://arxiv.org/pdf/1910.00138.pdf# -*- coding: utf-8 -*-

"""
Created on Wed May  4 14:02:58 2022

@author: nicol
"""
#%%
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.colors import LogNorm
from matplotlib.ticker import MultipleLocator


pathimg = r'C:\Users\nicol\Documents\Python\lulu\PortII\750.tiff'
image = cv2.imread(pathimg)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# class filter():
# Create a custom kernel

# src : https://alyxion.github.io/Udacity_IntroToSelfDrivingCarsNd/8_1_Intro_Mini_Projects/41_Finding%20Edges%20and%20Custom%20Kernels.html
# 3x3 array for edge detection
sobel_y = np.array([
                    [ -1, -2, -1], 
                    [  0,  0,  0], 
                    [  1,  2,  1]
                ])

## TODO: Create and apply a Sobel x operator
sobel_x = np.array([
                    [ -1, 0, 1], 
                    [ -2, 0, 2], 
                    [ -1, 0, 1]
                ])

# scharr sobel filter 
scharr_sobel = sobel_x+sobel_y*1j

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_y = cv2.filter2D(gray, -1, sobel_y)

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_x = cv2.filter2D(gray, -1, sobel_x)

fig1, axes = plt.subplots(2, 2, figsize=(15, 20), constrained_layout=False)
ax1 = axes[0, 0]
ax2 = axes[0, 1]
ax3 = axes[1, 0]
ax4 = axes[1, 1]

fig1.suptitle('Images filtered', fontsize=20)

ax1.set_title('sobel x')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im1 = ax1.imshow(filtered_image_x, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider1 = make_axes_locatable(ax1)
# Append axes to the right of ax1, with 20% width of ax1
cax1 = divider1.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar1 = plt.colorbar(im1, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax1
ax1.xaxis.set_visible(False)

ax2.set_title('sobel y')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im2 = ax2.imshow(filtered_image_y, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider2 = make_axes_locatable(ax2)
# Append axes to the right of ax2, with 10% width of ax2
cax2 = divider2.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar2 = plt.colorbar(im2, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax2
ax2.xaxis.set_visible(False)
ax2.yaxis.set_visible(False)



blur_filter = np.array([
                        [ 0.05, 0.2, 0.05], 
                        [  0.2, 0.2,  0.2], 
                        [ 0.05, 0.2, 0.05]
                    ])
blur_filter = np.array([
                        [ 0.005, 0.02, 0.005], 
                        [  0.02, 0.02,  0.02], 
                        [ 0.005, 0.02, 0.005]
                    ])

L = np.array([
    [  0,  0, -1,  0,  0],
    [  0, -1, -2, -1,  0],
    [ -1, -2,-16, -2, -1],
    [  0, -1, -2, -1,  0],
    [  0,  0, -1,  0,  0]
]) #best

edge_f = np.array([
                [-1,-1,-1],
                [-1,-8,-1],
                [-1,-1,-1]
            ])

def Normal_f(x,mu,sigma):
    G = np.exp((x-mu/sigma)**2*(-1/2))/(sigma*2*np.pi)
    return G

def Gab(c,sigma):
    m,n=(c,c)
    G = np.zeros((m,n))
    for a in range(m):
        for b in range(n):
            
            if c/2>int(c/2):
                G[a,b] = np.exp(((a-int((m)/2))**2+(b-int((n)/2))**2)*(-0.5)*(1/sigma)**2)#/sum(np.exp(((a-(m+1)/2)**2+(b-(n+1)/2)**2)*(-1/2)))
           
            else:
                pass
                G[a,b] = np.exp(((a-int((m+1)/2))**2+(b-int((n+1)/2))**2)*(-0.5)*(1/sigma)**2)#/sum(np.exp(((a-(m+1)/2)**2+(b-(n+1)/2)**2)*(-1/2)))
   
    return G/sum(sum(G))

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_blur = cv2.filter2D(gray, -1, blur_filter)

#------------------------------------------------------------------------------------
edge_detection = np.array([
                    [ 0.0, -1,  0], 
                    [  -1,  4, -1], 
                    [   0, -1,  0]
                ])

# Filter the image using filter2D, which has inputs: (grayscale image, bit-depth, kernel)  
filtered_image_edges = cv2.filter2D(filtered_image_blur, -1, edge_detection)

ax3.set_title('image_blur')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax3)
im3 = ax3.imshow(filtered_image_blur, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider3 = make_axes_locatable(ax3)
# Append axes to the right of ax3, with 30% width of ax3
cax3 = divider3.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar3 = plt.colorbar(im3, cax=cax3, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax3
ax3.xaxis.set_visible(False)

ax4.set_title('image_edges')
# Display image, `aspect='auto'` makes it fill the whole `axes` (ax4)
im4 = ax4.imshow(filtered_image_edges, norm=LogNorm(vmin=0.001, vmax=1), aspect='auto')
# Create divider for existing axes instance
divider4 = make_axes_locatable(ax4)
# Append axes to the right of ax4, with 40% width of ax4
cax4 = divider4.append_axes("right", size="20%", pad=0.05)
# Create colorbar in the appended axes
# Tick locations can be set with the kwarg `ticks`
# and the format of the ticklabels with kwarg `format`
cbar4 = plt.colorbar(im4, cax=cax4, ticks=MultipleLocator(0.2), format="%.2f")
# Remove xticks from ax4
ax4.yaxis.set_visible(False)
ax4.xaxis.set_visible(False)


#%%
# plot sauvage !!!
hey = Gab(10,1.5) ;x = np.linspace(0,9,10); plt.plot(x,hey);plt.grid();plt.title('gaussian_shape')

#%%
# Heat map on gray image
from scipy.ndimage import gaussian_filter;import matplotlib.pyplot as plt
result = gaussian_filter(gray, sigma=0.5)
plt.imshow(result, cmap='hsv')
cbar = plt.colorbar()
cbar.solids.set_edgecolor("face")
plt.draw()
#%%
# Gradient Magnitude and Gradient Orientation
# src : https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/#pyis-cta-modal
from scipy import signal
from scipy import misc

ascent = misc.ascent()
acsent = gray
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

grad = signal.convolve2d(gray, scharr, boundary='symm', mode='same')
import matplotlib.pyplot as plt
fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3, figsize=(15, 15))

ax_orig.imshow(gray, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_mag.imshow(np.absolute(grad), cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
ax_ang.set_title('Gradient orientation')
ax_ang.set_axis_off()

fig.show()

#%%
# filtre fin edge detecter, on voit les microstructure
# src : https://www.geeksforgeeks.org/python-edge-detection-using-pillow/
from PIL import Image,ImageFilter
import cv2

im  = Image.fromarray(cv2.imread(pathimg)) 
res = im.filter(ImageFilter.CONTOUR)
res.show('hsv')
final = im.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                 -1, -1, -1, -1), 1, 0))

final.show()
# plt.contour(np.reshape(im.getdata(),im.size))
#%%
# # plot 3D qui ne fonctionne pas
# from mpl_toolkits.mplot3d import Axes3D
# xe,ye = gray.shape
# X,Y = np.meshgrid(np.linspace(0,xe-1,xe),np.linspace(0,ye-1,ye))
# Z=np.asarray(gray).T

# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# def plot_figs(fig_num, elev, azim, Z):
#     fig = plt.figure(fig_num, figsize=(8, 8))
#     plt.clf()
#     ax = Axes3D(fig, elev=elev, azim=azim)

#     my_cmap= plt.get_cmap('cool')
#     surf = ax.plot_surface(
#         X, Y, Z,
#         alpha=0.3,
#         cmap = my_cmap,
#         edgecolor = 'none'
#     )
#     fig.colorbar(surf, ax = ax, shrink = 0.7 , aspect = 7)
#     plt.legend()

# # Generate the three different figures from different views
# elev = 43.5
# azim = -110
# plot_figs(1, elev, azim, 2)
# # %%
#%%
# Article à lire
# https://arxiv.org/pdf/1910.00138.pdf