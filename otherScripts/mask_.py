#%%           
import os, PIL
import numpy as np

from PIL import Image, ImageFilter # C:\Users\nicolas.casteleyn\Documents\py_Cobiss\00_Current_Codes\031221\python-3.8.1\python.exe -m pip install pillow
# from stackImages import stackImages, dirpath
from utils import dirpath

pathimg = os.path.join(dirpath(), r"700.0p5e-06.tiff")

originalImage = Image.open(pathimg) ; width, height = originalImage.size
originalImageArray=np.reshape(originalImage.getdata(),(-1,width))

minBitValue, maxBitValue = (0, 2**16-1)
minValueRef, maxValueRef = (3276, 20535) 
# minValueRef, maxValueRef = (3076, 15535) 
minValueRef, maxValueRef = (3076, 10535) 
minValue, maxValue = (minValueRef, maxValueRef)

print("minValue, maxValue", minValue, maxValue)

def masking(originalImageArray:np.ndarray, minValue:int, maxValue:int):
    Nlines, Ncolumns = np.shape(originalImageArray)
    maskImageArray = np.zeros_like(originalImageArray)

    for line in range(Nlines):
        for column in range(Ncolumns):
            bitValue = originalImageArray[line, column]
            if  bitValue <= maxValue and bitValue >= minValue:
                maskImageArray[line,column] = bitValue
            else:
                continue
    return maskImageArray

maskedImageArray = masking(originalImageArray, minValue, maxValue)
import matplotlib.pyplot as plt
plt.figure(1, figsize=(15,15))
plt.imshow(maskedImageArray, cmap='gray')

# print("originalImageArray", originalImageArray)
print("maskedImageArray", maskedImageArray)

maskedImage=Image.fromarray(maskedImageArray.astype(np.uint16))
# maskedImage.show()

# maskedImage.save("OLELEU.png") #fonctionne 
# maskedImage.save("OLELEU.tiff") #fonctionne pas
#%%           

# 
#       Filtering Technic 
# 
# https://pillow.readthedocs.io/en/stable/ 

# Gradient Magnitude and Gradient Orientation
# src : https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/#pyis-cta-modal
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter

gray = maskedImageArray
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

grad = signal.convolve2d(gray, scharr, boundary='symm', mode='same')

# plt.figure(2, figsize=(15,15))
fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3, figsize=(20, 20))

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

# sobel filter -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html#scipy.ndimage.sobel
# %%

# filter list
blur_filter = np.array([
                        [ 0.05, 0.2, 0.05], 
                        [  0.2, 0.2,  0.2], 
                        [ 0.05, 0.2, 0.05]
                    ])
blur_filter = np.array([
                        [ 0.005, 0.02, 0.005], 
                        [  0.02, 0.02,  0.02], 
                        [ 0.005, 0.02, 0.005]
                    ]) # <3

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
edge_f = np.array([
                [-1,-1,-1],
                [-1, 8,-1],
                [-1,-1,-1]
            ])
edge_f = np.array([
                [-0,-1,-0],
                [-1, 8,-1],
                [-0,-1,-0]
            ])

# Marr Hildreth Edge Detection
edge_detection = np.array([
                    [  0, -1,  0], 
                    [ -1,  4, -1], 
                    [  0, -1,  0]
                ])

# filtered_image_blur = signal.convolve2d(gray, blur_filter)
# filtered_image_edges = signal.convolve2d(filtered_image_blur, edge_detection)

filtered_image_blur = signal.convolve2d(gray, blur_filter, boundary='symm', mode='same')
filtered_image_edges = signal.convolve2d(filtered_image_blur, edge_detection, boundary='symm', mode='same')

plt.figure(3, figsize=(15,15))
fig, (ax_image_blur, ax_image_edges) = plt.subplots(1, 2, figsize=(15, 15))

ax_image_blur.imshow(filtered_image_blur, cmap='gray')
ax_image_blur.set_title('filtered_image_blur')
ax_image_blur.set_axis_off()

ax_image_edges.imshow(filtered_image_edges, cmap='gray')
ax_image_edges.set_title('filtered_image_edges')
ax_image_edges.set_axis_off()

#%%
gaussianImageArray = gaussian_filter(originalImageArray, sigma=3)
medianImageArray = median_filter(originalImageArray, size=20)

plt.figure(9, figsize=(15,15))
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

ax_orig = axes[0, 0]
ax_gauss = axes[0, 1]
ax_med = axes[1, 0]
ax_res = axes[1, 1]
ax_medOrigGauss = axes[0, 2]
ax_edgOrigGauss = axes[1, 2]
ax_mskEdgOrigGauss = axes[2, 0]
ax_a = axes[2, 1]
ax_b = axes[2, 2]

ax_orig.imshow(originalImageArray, cmap='gray')
ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_gauss.imshow(gaussianImageArray, cmap='gray')
ax_gauss.set_title('ax_gauss')
ax_gauss.set_axis_off()

ax_med.imshow(medianImageArray, cmap='gray')
ax_med.set_title('ax_med')
ax_med.set_axis_off()

originalGaussImageArray = originalImageArray // gaussianImageArray
ax_res.imshow(originalGaussImageArray, cmap='gray')
ax_res.set_title('originalGaussImageArray')
ax_res.set_axis_off()

medianOrigGaussArray = median_filter(originalGaussImageArray, size=2)
ax_medOrigGauss.imshow(medianOrigGaussArray, cmap='gray')
ax_medOrigGauss.set_title('medianOrigGaussArray')
ax_medOrigGauss.set_axis_off()

# edgeOrigGaussArray = signal.convolve2d(medianOrigGaussArray, edge_detection, boundary='symm', mode='same')
# edgeOrigGaussArray = signal.convolve2d(originalGaussImageArray, edge_f, boundary='symm', mode='same')
# edgeOrigGaussArray = signal.convolve2d(medianOrigGaussArray, edge_f, boundary='symm', mode='same')
edgeOrigGaussArray = signal.convolve2d(medianOrigGaussArray, L, boundary='symm', mode='same')
ax_edgOrigGauss.imshow(edgeOrigGaussArray, cmap='gray')
ax_edgOrigGauss.set_title('edgeOrigGaussArray')
ax_edgOrigGauss.set_axis_off()

minValue, maxValue =np.min(edgeOrigGaussArray) + 10*2, np.max(edgeOrigGaussArray)-0
print("minValue, maxValue",minValue, maxValue)
maskEdgeOrigGauss = masking(edgeOrigGaussArray, minValue, maxValue)
ax_mskEdgOrigGauss.imshow(maskEdgeOrigGauss, cmap='gray')
ax_mskEdgOrigGauss.set_title('maskEdgeOrigGauss')
ax_mskEdgOrigGauss.set_axis_off()

minValue, maxValue =np.min(edgeOrigGaussArray) + 10*4, np.max(edgeOrigGaussArray)-0
print("minValue, maxValue",minValue, maxValue)
a = masking(edgeOrigGaussArray, minValue, maxValue)
ax_a.imshow(a, cmap='gray')
ax_a.set_title('maskEdgeOrigGauss')
ax_a.set_axis_off()

minValue, maxValue =np.min(edgeOrigGaussArray) + 10*6, np.max(edgeOrigGaussArray)-0
print("minValue, maxValue",minValue, maxValue)
b = masking(edgeOrigGaussArray, minValue, maxValue)
ax_b.imshow(b, cmap='gray')
ax_b.set_title('maskEdgeOrigGauss')
ax_b.set_axis_off()

#%%
plt.figure(4, figsize=(15,15))
plt.imshow(np.abs(filtered_image_edges))

#%%
# filtered_image_edges = [ bit>0 for bit in filtered_image_edges]
maxValue = int(443 - 43)
maxValue = int(250)
filtered_image_edges = masking(filtered_image_edges, 0, maxValue)

minValue = np.min(filtered_image_edges)+50*2
# minValue = np.min(filtered_image_edges)
filtered_image_edges = masking(filtered_image_edges, minValue, maxValue)

plt.figure(5, figsize=(15,15))
print("minValue, maxValue", minValue, maxValue)
plt.imshow(filtered_image_edges)

# %%

#%%           

# 
#       Filtering Technic 
# 
# https://pillow.readthedocs.io/en/stable/ 

# Gradient Magnitude and Gradient Orientation
# src : https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/#pyis-cta-modal
from scipy import signal
from scipy.ndimage import gaussian_filter, median_filter

gray = originalGaussImageArray
scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

grad = signal.convolve2d(gray, scharr, boundary='symm', mode='same')

# plt.figure(2, figsize=(15,15))
fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(1, 3, figsize=(20, 20))

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

# sobel filter -> https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.sobel.html#scipy.ndimage.sobel
#%%      
TestedImage = originalImageArray     
# TestedImage = maskedImageArray 
    
Imin, Imax = np.min(TestedImage), np.max(TestedImage)

#conditionnal minimum:
Imin = np.min([i for i in TestedImage.flatten()[np.where(TestedImage.flatten() > 0)]])

(lines, columns), size = np.shape(TestedImage), np.size(TestedImage)
In = np.linspace(0,Imax, Imax+1)
labelhist = 'Imin: {}\nImax: {}\nlines: {}\ncolumns: {}\nsize: {}\n '.format(Imin,Imax, lines, columns, size)

plt.figure(99,figsize = (15,15))

plt.subplot(221)
plt.imshow(TestedImage,cmap='gray')

plt.subplot(222)
plt.hist(TestedImage.ravel(), bins=int(2**8), fc='k', ec='k', label = labelhist) #calculating histogram
plt.xlim((Imin, Imax)) #calculating histogram
# plt.ylim((0, Imax)) #calculating histogram
plt.legend()
plt.grid()
plt.xlabel('pixel intensity') #calculating histogram
plt.ylabel('number of pixels') #calculating histogram
plt.title('distrubtion of the number of pixels through their value')

# https://www.youtube.com/watch?v=mfWHI1fhpvc
# occurence 

unique, counts = np.unique(TestedImage, return_counts=True)
res = dict(zip(unique, counts))

fn = counts/size
foreach = [ sum(fn[:index]) for index, element in enumerate(fn)]
Fn = np.array(foreach)

plt.subplot(223)
label = 'frequency distribution fn'
plt.plot(unique,fn,'x', color= 'r',label = label ) #calculating histogram
plt.plot(unique,fn,label = label ) #calculating histogram
plt.grid()
plt.xlim((Imin, Imax)) #calculating histogram
plt.xlabel('pixel intensity') #calculating histogram
plt.ylabel(label) #calculating histogram
plt.title(label)

plt.subplot(224)
label = 'cumulative frequency distribution Fn'
plt.plot(unique,Fn,label = label ) #calculating histogram
plt.grid()
plt.xlabel('pixel intensity') #calculating histogram
plt.xlim((Imin, Imax)) #calculating histogram
plt.ylabel(label) #calculating histogram
plt.title(label)
#%%           
print(np.amin(maskedImageArray,initial=1))
#%%           
# https://pomain.medium.com/how-to-build-gif-video-from-images-with-python-pillow-opencv-c3126ce89ca8


# https://dev.to/slushnys/how-to-create-a-video-from-an-image-with-python-26p5
# --------------------------------------------------------
# https://realpython.com/python-import/#the-python-import-system

# https://realpython.com/python-gui-tkinter/

# https://docs.python.org/3/library/tkinter.html#mapping-basic-tk-into-tkinter

# https://tkdocs.com/tutorial/windows.html#createdestroy

# https://realpython.com/image-processing-with-the-python-pillow-library/#image-manipulation-with-numpy-and-pillow

# https://neptune.ai/blog/image-processing-python

 