#%%
from utils import dirpath, locateFileExt, buffer2dataTXT
import os , sys, time, PIL
import pandas as pd
# import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
from matplotlib import cm

pathimg = os.path.join(dirpath(), r"700.0p5e-06.tiff")

basepath = os.path.join(dirpath(), r"082522_P2C3-700-824nm\power5.0umP2C3_0.1nm_1000ms_700-824nm")
basepaths = locateFileExt(path = basepath, ext='.tiff')
#%%
lbdMIN,lbdMAX=750,824;step=1#nm
N=int(np.abs(lbdMAX-lbdMIN)/step+1)
wavelength_range=np.linspace(lbdMIN,lbdMAX,N)#nm    

#size of a nanostructered area in pixel number 
Zsize, ZsizeL=(21,21), (27,27)

#space dimension between zones
shift = (29,31)

#repear the square from the top-left corner #Z1=(672,442);Z2=(703,442);Z3=(734,442);Z4=(672,471);Z5=(672,500);Z6=(671,562)
Z={
'1':(672,442),

'2':(703,442),

'3':(734,442),

'4':(672,471),

'5':(671,501),

'6':(671,562),
}
#%%
#------------------------------------------------------------------------------
# For ncropped image
A = {"11":(609,412),"12":(640,412),"13":(671,412),"14":(702,412),"15":(733,412),
     "21":(609,441),"22":(640,441),"23":(671,441),"24":(702,441),"25":(733,441),
     "31":(609,471),"32":(640,471),"33":(671,472),"34":(702,471),"35":(733,471),
     "41":(609,501),"42":(640,501),"43":(671,501),"44":(702,501),"45":(733,501),
     "51":(609,530),"52":(640,530),"53":(671,530),"54":(702,530),"55":(733,530)}

#------------------------------------------------------------------------------
# position = 17
# O = {f"{a}":(A[a][0]-A[list(A.keys())[position]][0],A[a][1]-A[list(A.keys())[position]][1]) for a in list(A.keys())}
# Z5 = (302,322)
# Z5 = (323,300)
# Anew = {f"{a}":(O[a][0]+Z5[0],O[a][1]+Z5[1]) for a in list(O.keys())}
#------------------------------------------------------------------------------

# For cropped image
Anew = {
        '11': (240, 233), '12': (271, 233), '13': (302, 233), '14': (333, 233), '15': (364, 233),
        '21': (240, 262), '22': (271, 262), '23': (302, 262), '24': (333, 262), '25': (364, 262),
        '31': (240, 292), '32': (271, 292), '33': (302, 293), '34': (333, 292), '35': (364, 292),
        '41': (240, 322), '42': (271, 322), '43': (302, 322), '44': (333, 322), '45': (364, 322),
        '51': (240, 351), '52': (271, 351), '53': (302, 351), '54': (333, 351), '55': (364, 351)
    }
# Anew = {
#     '11': (261, 211),'12': (292, 211),'13': (323, 211),'14': (354, 211),'15': (385, 211),
#     '21': (261, 240),'22': (292, 240),'23': (323, 240),'24': (354, 240),'25': (385, 240),
#     '31': (261, 270),'32': (292, 270),'33': (323, 271),'34': (354, 270),'35': (385, 270),
#     '41': (261, 300),'42': (292, 300),'43': (323, 300),'44': (354, 300),'45': (385, 300),
#     '51': (261, 329),'52': (292, 329),'53': (323, 329),'54': (354, 329),'55': (385, 329)
#     }
#%%
im = Image.open(pathimg) ; width, height = im.size
im_array = np.array(im) 


#Totalité de la structures
nanostructuesArray = 243,240, 180,152

# #Z5 -Hband
y,x,h,w = 243,323, 29,152
 

# #Z5 -Vband
y,x,h,w = 302,240, 180,27

# #Z5
Z5 = (323,300)
# Z5 = (302,322)
y,x,h,w = Anew['41'][::-1] + ZsizeL
# y,x,h,w = Anew['41'] + ZsizeL

# ###visualization of the result image
# plt.figure(figsize=(15,15))
# plt.subplot(221)
# res_array = im_array[x:int(x+h),y:int(y+w)]
# plt.imshow(res_array.astype('uint16')) # cmap='gray')

# y,x,h,w = 243,240, 180,152
# imNew_array=np.array([im_array[x:int(x+h),y:int(y+w)].T,im_array[x:int(x+h),y:int(y+w)].T,im_array[x:int(x+h),y:int(y+w)].T]).T
imNew_array=np.array([im_array.T,im_array.T,im_array.T]).T
imNew = Image.fromarray((imNew_array*((2**8-1)/(2**16-1))).astype(np.uint8))

#fonctionne pas
# imNew_array=(imNew_array).astype(np.uint8)
# imNew = Image.fromarray((imNew_array*((2**8-1)/(2**16-1))).astype(np.uint16))

draw = ImageDraw.Draw(imNew)
shape = [(x, y), (int(x+h),int(y+w))]
draw.rectangle(shape, outline ="red")
plt.figure(figsize=(15,15))
plt.subplot(221)
# res_array = im_array[shape[0][0]:shape[1][0],shape[0][1]:shape[1][1]]
res_array = im_array[shape[0][1]:shape[1][1],shape[0][0]:shape[1][0]]
plt.imshow(res_array.astype('uint16')) # cmap='gray')

plt.subplot(222)
plt.imshow(imNew)

plt.subplot(224)
y,x,h,w = nanostructuesArray
imcroped_array = np.array([(np.array(imNew)[x:int(x+h),y:int(y+w),0]).T,(np.array(imNew)[x:int(x+h),y:int(y+w),1]).T,(np.array(imNew)[x:int(x+h),y:int(y+w),2]).T]).T
imcroped = Image.fromarray(imcroped_array)
plt.imshow(imcroped)

plt.subplot(223)
crop_array = np.array(np.array(imNew)[x:int(x+h),y:int(y+w),2])
crop = Image.fromarray(crop_array)
plt.imshow(crop)
#%%