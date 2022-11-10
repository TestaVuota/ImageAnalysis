#%%
from doctest import script_from_examples
from utils import dirpath, locateFileExt, heatmap, annotate_heatmap
import os , sys, time, PIL
import pandas as pd
# import seaborn as sns
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
# from  visualisation import A, shift, Zsize, ZsizeL, nanostructuesArray

basepath = dirpath()
basepath_ = os.path.join(dirpath(), r"082522_P2C3-700-824nm\power5.0umP2C3_0.1nm_1000ms_700-824nm")
basepath_ = r'Z:\SPARC\09_Measures\081722_P1C2\power5.0umP1C2_0.1'
basepath_ = r'Z:\SPARC\09_Measures\080822_P1C3\power5umP1C3_0.1'
basepath_ = r'Z:\SPARC\09_Measures\081022_P2C3-P1C2P2C3\power5umP2C3_0.1'


#Fait
basepath_ = r'Z:\SPARC\09_Measures\081922\power5.0umP1C3_0.1nm_1000ms'
basepath_ = r'Z:\SPARC\09_Measures\081922\power5.0umP2C2_0.1nm_1000ms'
basepath_ = r'Z:\SPARC\09_Measures\082922_P1C2P2C3-P2C3\power5.0umP1C2P2C3_0.1nm_1000ms_700-850nm'
# # ----
basepath_ = r'Z:\SPARC\09_Measures\082222\power5.0umP1C3P2C2_0.1nm_1000ms'
# basepath_ = r'Z:\SPARC\09_Measures\082322\power5.0umP2C3_0.1nm_1000ms'
# basepath_ = r'Z:\SPARC\09_Measures\082622_P1C2-P2C3\power5.0umP1C2_0.1nm_1000ms_700-850nm'


#A faire
# basepath_ = r'Z:\SPARC\09_Measures\082922_P1C2P2C3-P2C3\power5.0umP2C3_0.1nm_1000ms_700-850nm'

# basepath_ = r'Z:\SPARC\09_Measures\081022_P2C3-P1C2P2C3\power2.5umP1C2P2C3_0.1'
# basepath_ = r'Z:\SPARC\09_Measures\081122_P1C2P2C3-P1C3P2C2\power2.5umP1C3P2C2_0.1'


basepath_ = r'Z:\SPARC\09_Measures\081222_P2C2\power2.5umP2C2_0.1'
basepath_ = r'Z:\SPARC\09_Measures\081622_P2C2\power5.0umP2C2_0.1'

basepath_ = r'Z:\SPARC\09_Measures\081722_P1C2\power2.5umP1C2_0.1'
# basepath_ = r'Z:\SPARC\09_Measures\081722_P1C2\power5.0umP1C2_0.1'

basepaths = locateFileExt(path = basepath_, ext='.tiff')
#%%
basepath = basepaths[-1];print(basepath)
im = Image.open(basepath) ; width, height = im.size
im_array = np.array(im) 
#%% 
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
areaLabel = '43'
cropCoordinates = A[areaLabel] + ZsizeL
y,x,h,w = cropCoordinates
res_array = im_array[x:int(x+h),y:int(y+w)]
imres = Image.fromarray(res_array.astype('uint16'))

# ax_aimedImg
# Definition dune matrice RGB avec pour base l image SPARC
imNew_array=np.array([im_array.T,im_array.T,im_array.T]).T
imNew = Image.fromarray((imNew_array*((2**8-1)/(2**16-1))).astype(np.uint8))

# Dessin dun rectangle rouge au coordonnée de la zoe A['']
draw = ImageDraw.Draw(imNew)
shape = [(y, x), (int(y+w),int(x+h))]
draw.rectangle(shape, outline ="red")

# Rognage de l image resultant aux bornes de la zone d interêt
y,x,h,w = nanostructuesArray
imcroped_array = np.array([(np.array(imNew)[x:int(x+h),y:int(y+w),0]).T,(np.array(imNew)[x:int(x+h),y:int(y+w),1]).T,(np.array(imNew)[x:int(x+h),y:int(y+w),2]).T]).T
imcroped = Image.fromarray(imcroped_array)

fig, axes = plt.subplots(1, 2, figsize=(15, 15))
ax_aimedImg = axes[0]
ax_crop = axes[1]

ax_aimedImg.imshow(imcroped, cmap='gray')
ax_aimedImg.set_title('A{} \n{}'.format(areaLabel,basepath.split('\\')[-1].split('.tiff')[0]))
ax_aimedImg.set_axis_off()

ax_crop.imshow(imres, cmap='gray')
ax_crop.set_title('A{}'.format(areaLabel))
ax_crop.set_axis_off()
#%%
stackImage = []
for a in list(A.values()):
    IM_STACK = []
    for basepath in basepaths[:]:
        im = Image.open(basepath) ; width, height = im.size#;print('im.info',im.info) 
        # im_array = np.reshape(im.getdata(),(-1,width)) 
        im_array = np.array(im) 
            
        y,x,h,w = a + ZsizeL
        xs,xe,ys,ye = x,int(x+h),y,int(y+w) 
        res_array = im_array[xs:xe,ys:ye].astype('uint16')
        
        IM_STACK.append(res_array.flatten())
    stackImage.append(IM_STACK)
# np.shape(stackImage) -> (labelStructuredArea, nImages, nPixels) -> (25, 2, 729)
# IM_ARRAY = np.asarray(IM_STACK).T
#%
np.save(os.path.join(dirpath(),'previous_exp\\stackImage{}.npy'.format(basepath_.split('\\')[-1])),stackImage)
#%%



#%%
from scipy.optimize import nnls #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
from scipy.optimize import lsq_linear #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear
Aimages = {f"{list(A.keys())[i]}":image for i, image in enumerate(stackImages)}
steps=(1,2,3,4,5,8,10,12)
test_array = Aimages['43'][12,:]

###NON NEGATIVE LEAST SQUARE METHOD
plt.figure(1,figsize=(15,15))
for step in steps:
    learn_array = Aimages['43'][::step,:].T
    IM_STACK,R = learn_array , test_array
    X , rnorm = nnls(IM_STACK,R) #with X the solution vector and rnorm the residual norm
    res = lsq_linear (IM_STACK,R)
    lbdMIN,lbdMAX=700,824;step=float(.1*step)#nm
    N=int(np.abs(lbdMAX-lbdMIN)/step+1)
    wavelength_range=np.linspace(lbdMIN,lbdMAX,N)#nm 
    label = f'step : {step}'
    try:
        plt.plot(wavelength_range[:-1],X/np.abs(max(X)-min(X)),label = label)
    except:
        plt.plot(wavelength_range[:],X/np.abs(max(X)-min(X)),label = label)
plt.xlabel('wavelength in (nm)')
plt.ylabel("Normalized Intensity (I)")
plt.legend()
plt.grid()
#%%

# end of the working script 
# the rest of the code is for true test data
# which have not be done experimentally

#-------------------------------------------------------------------------#
# SRC: https://dev.to/miku86/vscode-transform-text-to-lowercase-uppercase-titlecase-1mcg
#-------------------------------------------------------------------------#
# Select the text to transform. Use Ctrl + L to selected the whole line
# Open Show all commands. Linux and Windows: Ctrl + Shift + P, Mac: ⇧⌘P
# Type in the command, e.g. lower, upper, title
# Wait for auto-complete (like in the animation)
# Hit Enter

#%%





























































#%% ###NON NEGATIVE LEAST SQUARE METHOD using scipy
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html

  ### DOWNLOADING DATAS of ONE SPECTRUM IMAGE composed of several WAVELENGTH PEAKS
pathimg = r'C:\Users\nicolas.casteleyn\Documents\py_Cobiss\09_Measures\020324\Mesures_multispectrales\PORTII\SP4\SP4.tiff'
testim = Image.open(pathimg)
testim_array = np.reshape(testim.getdata(),(-1,width))

test_array = testim_array[xs:xe,ys:ye].flatten()

###NON NEGATIVE LEAST SQUARE METHOD
from scipy.optimize import nnls #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.nnls.html
from scipy.optimize import lsq_linear #https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.lsq_linear.html#scipy.optimize.lsq_linear
IM_STACK,R = IM_ARRAY , test_array
X , rnorm = nnls(IM_STACK,R) #with X the solution vector and rnorm the residual norm
res = lsq_linear (IM_STACK,R)
#------------------------------------------------------------------------------
paths = locateFileExt(path = os.path.dirname(pathimg), ext=".TXT")
txtnames = [s.split('\\')[-1] for s in paths]

              
plt.figure(2,figsize=(8,8))
for p in enumerate(paths[:]):
    plt.figure(p[0])
    A=pd.read_csv(p[1],skiprows=23,delimiter=',').to_numpy()
    x,y=A[:,0],A[:,1]
    label = txtnames[p[0]].split('.')[0].split('SAV ' )[1]

    plt.plot(x,y/np.abs(max(y)-min(y)),'x-',label=label)
    
    plt.plot(wavelength_range,X/np.abs(max(X)-min(X)),label = 'result from nnlsqs')
    plt.xlabel("wavelength")
    plt.ylabel("Normalized_Intensity")
    plt.grid();plt.legend()    
    plt.title("NNLSQS")
    
    figname = label+".png"
    savepath= r'C:\Users\nicolas.casteleyn\Documents\py_Cobiss\11_Image_Analysis\1_IMG_Yassine'
    plt.savefig(os.path.join(savepath,figname)) 
    
#%% MACHINE LEARNING WITH LINEAR REGRESSION
# NNLS == https://scikit-learn.org/stable/auto_examples/linear_model/plot_nnls.html
#----------------------------------------------------------------------------------------------------------------

#Choosing the model of regression

#NNLSQS
from sklearn.linear_model import LinearRegression  ;  model = LinearRegression(positive=True)

#LSQS
# from sklearn.linear_model import LinearRegression  ;  model = LinearRegression(positive=False)

#LSQS l1-norm
# from sklearn.linear_model import Lasso             ;  model = Lasso(alpha=0.1) ##Y=f(X)=f(X1,X2,..,Xn)

#LSQS l2-norm
# from sklearn.linear_model import Ridge             ;  model = Ridge(alpha = 1.0)
#----------------------------------------------------------------------------------------------------------------

# R = test_array
# fit = model.fit(IM_STACK,R)
# coef,intercept = model.coef_, model.intercept_

# predict = model.predict(test_array.reshape(-1,R.size)) ; print(predict)

# plt.figure(3,figsize=(8,8))
# for p in enumerate(paths[:]):
#     plt.figure(p[0])
#     A=pd.read_csv(p[1],skiprows=23,delimiter=',').to_numpy()
#     x,y=A[:,0],A[:,1]
#     label = txtname[p[0]].split('.')[0].split('SAV ' )[1]

#     plt.plot(x,y/np.abs(max(y)-min(y)),'x-',label=label)
    
#     plt.plot(wavelength_range,coef/np.abs(max(coef)-min(coef)),label = 'result from nnlsqs')
#     plt.xlabel("wavelength")
#     plt.ylabel("Normalized_Intensity")
#     plt.grid();plt.legend()    
#     plt.title("NNLSQS")
    
#     figname = label+".png"
#     savepath= r'C:\Users\nicolas.casteleyn\Documents\py_Cobiss\11_Image_Analysis\1_IMG_Yassine'
#     plt.savefig(os.path.join(savepath,figname)) 
# #%%