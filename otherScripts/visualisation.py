#%%
from utils import (
                    dirpath,locateFileExt,
                    # heatmap,annotate_heatmap,
                    masking)
import os , sys, time, PIL
import pandas as pd
# import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import matplotlib.pyplot as plt
size = 3
Filters = [
            ImageFilter.BLUR,
            ImageFilter.CONTOUR,
            ImageFilter.DETAIL,
            ImageFilter.EDGE_ENHANCE,
            ImageFilter.EDGE_ENHANCE_MORE,
            ImageFilter.EMBOSS,
            ImageFilter.FIND_EDGES,
            ImageFilter.SHARPEN,
            ImageFilter.SMOOTH,
            ImageFilter.SMOOTH_MORE,
            ImageFilter.GaussianBlur,
            ImageFilter.MedianFilter,
            ImageFilter.MaxFilter,
            ImageFilter.MinFilter,
            ImageFilter.Kernel(
                                (3, 3),
                                (
                                    -1, -1, -1,
                                    -1,  8, -1, 
                                    -1, -1, -1
                                ),
                                1,
                                0
                            ),
            ImageFilter.BoxBlur(radius = 3),
            ImageFilter.UnsharpMask(radius=3, percent=150, threshold=3),
            ImageFilter.ModeFilter(size=3),
            # ImageFilter.Kernel(
            #                     (3, 3),
            #                     (
            #                         -0, -1, -0,
            #                         -1,  4, -1, 
            #                         -0, -1, -0,
            #                     ),
            #                     1,
            #                     0
            #                 ),
            ImageFilter.RankFilter(size = size, rank = int(size**3 / (3*2))),
            ImageFilter.BoxBlur(radius = 0),
            # size – The kernel size, in pixels.
            # rank – What pixel value to pick.
            # Use 0 for a min filter,
            # size * size / 2 for a median filter,
            # size * size - 1 for a max filter, etc.
            # ImageFilter.Color3DLUT,
        ]
basepath = r'Z:\SPARC\09_Measures\081722_P1C2\power5.0umP1C2_0.1'
basepath =  dirpath()
basepaths = locateFileExt(path = basepath, ext='.tiff')

basepath = basepaths[-1]
im = Image.open(basepath) ; width, height = im.size
im_array = np.array(im) 
#%% 
# Custom_Filter

scharr = np.array([
                    [ -3-3j, 0-10j,  +3 -3j],
                    [-10+0j, 0+ 0j, +10 +0j],
                    [ -3+3j, 0+10j,  +3 +3j]
                ]) # Gx + j*Gy

sobel_y = np.array([
                    [ -1, -2, -1], 
                    [  0,  0,  0], 
                    [  1,  2,  1],
                ])

prewitt_x = np.array([
                    [ 1, 0, -1], 
                    [ 1, 0, -1], 
                    [ 1, 0, -1],
                ])
prewitt_y = prewitt_x.T
## TODO: Create and apply a Sobel x operator
sobel_x = sobel_y.T

# scharr sobel filter 
scharr_sobel = sobel_x+sobel_y*1j

# sobel filter X-axis
sobel = sobel_x * -1/8

# prewitt filter X-axis
prewitt = np.array([
                    [ 1, 0, -1], 
                    [ 1, 0, -1], 
                    [ 1, 0, -1]
                ]) * 1/6

# Marr-Hildreth filter X-axis
marr_hildreth = np.array([
                    [ 0, 1, 0], 
                    [ 1, 4, 1], 
                    [ 0, 1, 0]
                ]) 

edge_detection = np.array([
                    [   0, -1,  0], 
                    [  -1,  4, -1], 
                    [   0, -1,  0],
                ])

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
#%% https://neptune.ai/blog/image-processing-python-libraries-for-machine-learning
#size of a nanostructered area in pixel number 
Zsize, ZsizeL=(21,21), (27,27)

#space dimension between zones
shift = (29,31)

#
# For cropped image
A = {
        '01': (242, 233), '02': (274, 233), '03': (305, 233), '04': (336, 233), '05': (367, 233),
        '11': (242, 262), '12': (274, 262), '13': (305, 262), '14': (336, 262), '15': (367, 262),
        '21': (242, 294), '22': (274, 294), '23': (305, 294), '24': (336, 294), '25': (367, 294),
        '31': (242, 324), '32': (274, 324), '33': (305, 324), '34': (336, 324), '35': (367, 324),
        '41': (242, 354), '42': (274, 354), '43': (305, 354), '44': (336, 354), '45': (367, 354),
        '51': (242, 383), '52': (274, 383), '53': (305, 383), '54': (336, 383), '55': (367, 383),
        '61': (242, 412), '62': (274, 412), '63': (305, 412), '64': (336, 412), '65': (367, 412),
    }
Z = {   
        '_01': (242, 233), '_02': (274, 233), '_03': (305, 233), '_04': (336, 233), '_05': (367, 233),
        '_11': (242, 262), '_12': (274, 262), '_13': (305, 262), '_14': (336, 262), '_15': (367, 262),
        '_21': (242, 294), '_22': (274, 294),  '1': (305, 294),  '2': (336, 294),  '3': (367, 294),
        '13': (242, 324), '11': (274, 324),  '4': (305, 324),  '7': (336, 324),  '9': (367, 324),
        '14': (242, 354), '12': (274, 354),  '5': (305, 354),  '8': (336, 354), '10': (367, 354),
        '15': (242, 383), '16': (274, 383),  '6': (305, 383), '17': (336, 383), '18': (367, 383),
        '_61': (242, 412), '_62': (274, 412), '_63': (305, 412), '_64': (336, 412), '_65': (367, 412),
    }  

#Totalité de la structures
nanostructuesArray = 243,242, 180,152

# ax_crop
areaLabel = '23'
areaLabel = '13'
areaLabel = '33'
areaLabel = '43'
# areaLabel = '53'
# areaLabel = '63'
cropCoordinates = A[areaLabel] + ZsizeL
y,x,h,w = cropCoordinates
res_array = im_array[x:int(x+h),y:int(y+w)]
imres = Image.fromarray(res_array.astype('uint16'))

# ax_aimedImg
# Definition dune matrice RGB avec pour base l image SPARC
imNew_array=((np.array([im_array.T,im_array.T,im_array.T]).T)*((2**8-1)/(2**16-1))).astype(np.uint8)
imNew = Image.fromarray(imNew_array)

# Dessin dun rectangle rouge au coordonnée de la zoe A['']
draw = ImageDraw.Draw(imNew)
shape = [(y, x), (int(y+w),int(x+h))]
draw.rectangle(shape, outline ="red")

# Rognage de l image resultant aux bornes de la zone d interêt
y,x,h,w = nanostructuesArray
imcroped_array = np.array([(np.array(imNew)[x:int(x+h),y:int(y+w),0]).T,(np.array(imNew)[x:int(x+h),y:int(y+w),1]).T,(np.array(imNew)[x:int(x+h),y:int(y+w),2]).T]).T
imcroped = Image.fromarray(imcroped_array)

fig, axes = plt.subplots(3, 4, figsize=(15, 15))
ax_aimedImg = axes[0, 0]
ax_crop = axes[0, 1]
ax_mag = axes[0, 2]
ax_ang = axes[0, 3]
ax_sobelMagX = axes[1, 0]
ax_sobelAngX = axes[1, 1]
ax_prewittMagX = axes[1, 2]
ax_prewittAngX = axes[1, 3]
ax_sobelMagY = axes[2, 0]
ax_sobelAngY = axes[2, 1]
ax_prewittMagY = axes[2, 2]
ax_prewittAngY = axes[2, 3]

ax_aimedImg.imshow(imcroped, cmap='gray')
ax_aimedImg.set_title('A{} \n{}'.format(areaLabel,basepath.split('\\')[-1].split('.tiff')[0]))
ax_aimedImg.set_axis_off()

ax_crop.imshow(imres, cmap='gray')
ax_crop.set_title('A{}'.format(areaLabel))
ax_crop.set_axis_off()
#%
# Gradient Magnitude and Gradient Orientation
# src : https://pyimagesearch.com/2021/05/12/image-gradients-with-opencv-sobel-and-scharr/#pyis-cta-modal
from scipy import signal
from scipy import misc

a, b = 3, 10
# a, b = 1, 15
scharr = np.array([
                    [ -a-a*1j, 0-b*1j,  +a-a*1j],
                    [ -b+0*1j, 0+0*1j,  +b+0*1j],
                    [ -a+a*1j, 0+b*1j,  +a+a*1j],
                ]) # Gx + j*Gy

grad = signal.convolve2d(res_array, scharr, boundary='symm', mode='same')
gradSobelX = signal.convolve2d(grad, sobel_x, boundary='symm', mode='same')
gradPrewittX = signal.convolve2d(grad, prewitt_x, boundary='symm', mode='same')
gradSobelY = signal.convolve2d(grad, sobel_y, boundary='symm', mode='same')
# gradPrewittY = signal.convolve2d(grad, prewitt_y, boundary='symm', mode='same')


# gradPrewittY = signal.correlate2d(res_array, res_array[12:15,12:15], boundary='symm', mode='same')

# https://thispointer.com/apply-a-function-to-every-element-in-numpy-array/
def applyToMatrixElement(el,c=0):
    
    if type(el) == 'int':
        return int(round(el*2**-16))
    if type(el) == 'float':
        return float(round(el*2**-16,c))
    if type(el) == 'complex':
        return complex(round(el.real*2**-16,c),round(el.imag*2**-16,c))

gradPrewittY = signal.correlate2d(res_array, grad, boundary='symm', mode='same')
gradPrewittY = np.array(list(map(applyToMatrixElement, gradPrewittY)))


ax_mag.imshow(np.absolute(grad), cmap='gray')
ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles
ax_ang.set_title('Gradient orientation')
ax_ang.set_axis_off()

ax_sobelMagX.imshow(np.absolute(gradSobelX), cmap='gray') # hsv is cyclic, like angles
ax_sobelMagX.set_title('Gradient magnitude SobelX')
ax_sobelMagX.set_axis_off()

ax_prewittMagX.imshow(np.absolute(gradPrewittX), cmap='gray') # hsv is cyclic, like angles
ax_prewittMagX.set_title('Gradient magnitude PrewittX)')
ax_prewittMagX.set_axis_off()

ax_sobelAngX.imshow(np.angle(gradSobelX), cmap='hsv') # hsv is cyclic, like angles
ax_sobelAngX.set_title('Gradient orientation SobelX')
ax_sobelAngX.set_axis_off()

ax_prewittAngX.imshow(np.angle(gradPrewittX), cmap='hsv') # hsv is cyclic, like angles
ax_prewittAngX.set_title('Gradient orientation PrewittX')
ax_prewittAngX.set_axis_off()

ax_sobelMagY.imshow(np.absolute(gradSobelY), cmap='gray') # hsv is cyclic, like angles
ax_sobelMagY.set_title('Gradient magnitude SobelY')
ax_sobelMagY.set_axis_off()

#%%
ax_prewittMagY.imshow(np.absolute(gradPrewittY), cmap='gray') # hsv is cyclic, like angles
ax_prewittMagY.set_title('Gradient magnitude PrewittY')
ax_prewittMagY.set_axis_off()

ax_sobelAngY.imshow(np.angle(gradSobelY), cmap='hsv') # hsv is cyclic, like angles
ax_sobelAngY.set_title('Gradient orientation SobelY')
ax_sobelAngY.set_axis_off()
# ax_prewittAngY.imshow(np.angle(gradPrewittY), cmap='hsv') # hsv is cyclic, like angles
ax_prewittAngY.imshow(gradPrewittY, cmap='gray') # hsv is cyclic, like angles
ax_prewittAngY.set_title('Gradient orientation PrewittY')
ax_prewittAngY.set_axis_off()

#%%
# stackImage = []
# for a in list(A.values()):
#     IM_STACK = []
#     for basepath in basepaths[:]:
#         im = Image.open(basepath) ; width, height = im.size#;print('im.info',im.info) 
            
#         y,x,h,w = a + ZsizeL
#         xs,xe,ys,ye = x,int(x+h),y,int(y+w) 
#         res_array = im_array[xs:xe,ys:ye].astype('uint16')
        
#         IM_STACK.append(res_array.flatten())
#     stackImage.append(IM_STACK)
# # np.shape(stackImage) -> (labelStructuredArea, nImages, nPixels) -> (25, 2, 729)
# # IM_ARRAY = np.asarray(IM_STACK).T
# stackImages = np.save(os.path.join(dirpath(),'stackImage.npy'),stackImage)

# #%%
# # Schema representation of the nanostructued Array
# width=5
# A_Mean = np.average(np.array(stackImage)[:,0,:], axis=1).reshape((int(len(A)/width),width))
# row_labels= ['{}'.format(i) for i in range(0,int(len(A)/width))]
# col_labels  = row_labels[1:-1]

# plt.figure(figsize=(15,15))
# fig, ax = plt.subplots()
# im, cbar = heatmap(A_Mean, row_labels, col_labels, ax=ax,
#                    cmap="YlGn", cbarlabel="mean of the pixel value per area")
# # texts = annotate_heatmap(im, valfmt="{x:.1f} t")
# texts = annotate_heatmap(im, valfmt="\n{x:.0f}")

# fig.tight_layout()
# plt.title('Nanostructured Areas Matrix A ')
# plt.show()

#%% 
nanostructuesArray = 243,240, 180,152
y,x,h,w = nanostructuesArray
y,x,h,w = cropCoordinates

res_array_ = im_array[x:int(x+h),y:int(y+w)]
im = Image.fromarray((res_array_*((2**8-1)/(2**16-1))).astype(np.uint8))

#%%

# All Filters from PIL - pillow 
from math import ceil
plotWidth, count = 4, 0

fig, axes = plt.subplots(ceil(len(Filters)/plotWidth), plotWidth, figsize=(20, 20))
shapeAxes = np.shape(axes)
immStack = []
for nl in range(shapeAxes[0]):
    for nc in range(shapeAxes[1]): 
        ax_im = axes[nl,nc]
        imm = im.filter(Filters[count])
        ax_im.imshow(imm, cmap='gray')
        ax_im.set_title(str(Filters[count]))
        ax_im.set_axis_off()
        count += 1
        immStack.append(imm)
#%%
im1 = im.filter(ImageFilter.BLUR)

im2 = im1.filter(ImageFilter.MinFilter(5))
im3 = (im.filter(ImageFilter.SMOOTH_MORE)).filter(ImageFilter.MinFilter)  # same as MinFilter(3)
im4 = Image.fromarray(np.array(im2)+np.array(im3))

im4_array = np.array(im4)
thresholdValue = ceil((np.max(im4_array)-np.min(im4_array)/2) - 3 )
im4N_array = masking(im4_array,thresholdValue,np.max(im4_array))
im0 = Image.fromarray(im4N_array).filter(ImageFilter.FIND_EDGES).filter(ImageFilter.SHARPEN)

im5_array =  np.array(im.filter(ImageFilter.MinFilter)) + np.array(im.filter(ImageFilter.SMOOTH_MORE)) + np.array(im.filter(ImageFilter.SHARPEN))*0 +np.array(im.filter(ImageFilter.BLUR)) + np.array(im.filter(ImageFilter.GaussianBlur))
im5 = Image.fromarray(im5_array)
#%
plt.figure(9, figsize=(15,15))
fig, axes = plt.subplots(3, 2, figsize=(15, 15))
ax_im = axes[0,0]
ax_im1 = axes[0,1]
ax_im2 = axes[1,0]
ax_im3 = axes[1,1]
ax_im0 = axes[2,0]
ax_im4 = axes[2,1]

ax_im.imshow(im, cmap='gray')
# ax_im.imshow(im.filter(ImageFilter.FIND_EDGES), cmap='gray')
ax_im.set_title('ax_im')
ax_im.set_axis_off()

ax_im1.imshow(im1, cmap='gray')
# ax_im1.imshow(im1.filter(ImageFilter.FIND_EDGES), cmap='gray')
ax_im1.set_title('ax_im1')
ax_im1.set_axis_off()

# ax_im2.imshow(im2, cmap='gray')
ax_im2.imshow(im2.filter(ImageFilter.FIND_EDGES), cmap='gray')
ax_im2.set_title('ax_im2')
ax_im2.set_axis_off()

# ax_im3.imshow(im3, cmap='gray')
ax_im3.imshow(im3.filter(ImageFilter.FIND_EDGES), cmap='gray')
ax_im3.set_title('ax_im3')
ax_im3.set_axis_off()


ax_im0.imshow(im5, cmap='gray')
# ax_im0.imshow(im5.filter(ImageFilter.FIND_EDGES), cmap='gray')
ax_im0.set_title('ax_im0')
ax_im0.set_axis_off()

# ax_im4.imshow(im4, cmap='gray')
ax_im4.imshow(im4.filter(ImageFilter.FIND_EDGES), cmap='gray')
ax_im4.set_title('ax_im4')
ax_im4.set_axis_off()

# %%
