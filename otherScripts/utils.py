import sys, os, ctypes
import numpy as np
import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt

def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw=None, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (M, N).
    row_labels
        A list or array of length M with the labels for the rows.
    col_labels
        A list or array of length N with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=("black", "white"),
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def buffer2dataTXT(pathbuffer,width):
    import pandas as pd;import numpy as np;import PIL ; from PIL import Image
    buffer=pd.read_csv(pathbuffer,delimiter=' ',header = None).to_numpy()
    buffer_array = np.reshape(buffer,(-1,width))
    data = np.transpose((buffer_array*((2**16-1)/(2**12-1)))).astype('uint16')
    
    return data

def lineEditReader(content: str = '', mainsep: str = ',', sepstart: str = '[', sepstop: str = ']') -> list:
    """
    content = "473,461,[465,755,10],1,21[300,793,10],19,18,[100,921,11]"
    > 473,461 is read as 473 -> 461
    > [465,755,10] is read as np.linspace(vmin=465,vmax=755,N) 
                        with N=int(np.abs(vmax-vmin)/step+1)
    """ 
    lspot = [] 
    if content.__contains__(sepstart) and  content.__contains__(sepstop):
            
        if content.count(sepstart) == content.count(sepstop):
            
            [lspot.append(item) for item in content.split(sepstart)[0].strip(mainsep).split(mainsep)]

            for i in range(1,content.count(sepstart)+1):
                sweep, spot = content.split(sepstart)[i].split(sepstop)

                start, stop, step = [int(i) for i in sweep.split(mainsep)]
                r = [str(val) for val in interval(start, stop, step)]
                lspot = lspot + r

                if spot != "":
                    [lspot.append(item) for item in spot.strip(mainsep).split(mainsep)]
                else:
                    pass
            lspot.pop(0)
        else:
            print("error on [,]")

    else:
        lsweep, lspot = None, []

        if content.__contains__(mainsep):
            [lspot.append(float(item)) for item in content.split(mainsep)]
       
        else:
            lspot = float(content)

    return  lspot


def callObject(idobj):
    callObj = ctypes.cast(idobj, ctypes.py_object).value
    return callObj

def dirpath():
    """
    C:\\Users\\nicol\\Documents\\Python\\pyCobiss -> path before the folder containing the application
    """
    return os.path.dirname(os.path.realpath(__file__))

def interval(vmin: float = 0,vmax: float = 10,step: float = 1) -> np.ndarray :
    N=int(np.abs(vmax-vmin)/step+1)
    vector = np.linspace(vmin,vmax,N)
    return vector

def temps(tstart: float = 0,tstop: float = 10,tsampling: float = 1) -> np.ndarray:
    return interval(tstart,tstop,tsampling)


def dictPrinter(param: dict = {}) -> None:
    [print(f"\n{key} \n\t {param[key]}") for key in list(param.keys())]

def dictReader(param: dict = {}) -> None:
    dictPrinter(param)
   

def filepath(filename: str = "") -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)),filename)

def locateUi(path) -> list:
    """ 
    Find uifile in the file directory
    """
    return locateFileExt(path=path,ext=".ui")
#---------------------------------------------------------------------------- 
def makeFileExt(content: str = "test",savepath: str = os.path.dirname(os.path.realpath(__file__)), name: str = "test", ext: str = ".txt"):
    try:
        if ext == ".json":
            content = json.dumps(content)
        else:
            pass
        with open(os.path.join(savepath,f'{name}{ext}'),'w') as f:
            f.write(content)
        print(f.closed)
    except:
        print("makeFileExt failed")

def readFileExt(dirpath: str = os.path.dirname(os.path.realpath(__file__)), name: str = "test", ext: str = ".txt") -> str:
    try:
        with open(os.path.join(dirpath,f'{name}{ext}'),'r') as f:
            content = f.read()
            if ext == '.json':
                data = json.loads(content)
                content = json.dumps(data, indent=4, sort_keys=True)
            else:
                data = None
        print(f.closed)
        return content, data
    except:
        print("readFileExt failed")

def locateFileExt(path: str = os.path.dirname(os.path.realpath(__file__)), ext: str = "") -> list:
    """ 
    Find file.ext in path 
    by default path is the file directory
    """
    paths, filename = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext):
                paths.append(os.path.join(root,file))
                filename.append(file)
    return paths

def locateFileNameExt(path: str = os.path.dirname(os.path.realpath(__file__)), containInName: str = "", ext: str = "") -> list:
    """ 
    Find file.ext in path 
    by default path is the file directory
    """
    paths, filename = [], []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(ext) and file.__contains__(containInName):
                paths.append(os.path.join(root,file))
                filename.append(file)
    paths.sort()
    return paths
    
def dirpath():
    """
    C:\\Users\\nicol\\Documents\\Python\\pyCobiss -> path before the folder containing the application
    """
    return os.path.dirname(os.path.realpath(__file__))

#------------------------------------------------------------------------------------------------------------------------#
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 16:09:31 2021

@author: nicol
"""
import cv2
import numpy as np
#%%
# https://docs.python.org/3/reference/datamodel.html
# findColorBIS.__name__
# findColorBIS.__doc__
#%%
def stackImages(scale,imgArray):
    rows, cols= len(imgArray),len(imgArray[0])
    rowsAvailable=isinstance(imgArray[0],list)
    width, height= imgArray[0][0].shape[1],imgArray[0][0].shape[0]
    
    if rowsAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                if imgArray[x][y].shape[:2]==imgArray[0][0].shape[:2]:
                    imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale) 
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                    # imgArray[x][y]=cv2.resize(imgArray[x][y],(0,0),None,scale,scale) #Faux
                if len(imgArray[x][y].shape)==2: 
                    imgArray[x][y]=cv2.cvtColor(imgArray[x][y],cv2.COLOR_GRAY2BGR)
                    
            imageBlank=np.zeros((height,width,3),np.uint8)
            hor=[imageBlank]*rows
            hor_con =[imageBlank]*rows            
        for x in range(0,rows):   
            hor[x]=np.hstack(imgArray[x])
        ver=np.vstack(hor)
            
    else:
        for x in range(0,rows):
            if imgArray[x].shape[:2]==imgArray[0][0].shape[:2]:
                imgArray[x]=cv2.resize(imgArray[x],(0,0),None,scale,scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
                # imgArray[x]=cv2.resize(imgArray[0][0] ,(0,0),None,scale,scale)
            if len(imgArray[x].shape)==2:
                imgArray[x]=cv2.cvtColor(imgArray[x],cv2.COLOR_GRAY2BGR)
                
        hor=np.hstack(imgArray)
        ver=hor           
    return ver
#%%
def getContours(img):
    #convert image to grayscale
    imgGray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur=cv2.GaussianBlur(imgGray,(7,7),1)
    #find the edges of our detector
    imgCanny=cv2.Canny(imgBlur,50,50)
    imgContours=img
    
    
    #the image + detection method + filtering contours for alter details
    #take the extrem out contours, and list it
    contours, hierarchy=cv2.findContours(imgCanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    
    # Do not put this line in the function
    # imgContours=img.copy()
    
    for cnt in contours:
        #the area of the contours
        area = cv2.contourArea(cnt)
        #draw the contour
        print('area',area)
        #img_src+targeted contour+-1 value to get all the contours,
        # colors of the contours + def img
        # cv2.drawContours(imgContours,cnt,-1,(255,0,0),3)
     #---------------WE COULD STOP HERE-----------------------   
        #defintion of a threshold area
        if area>500:
            cv2.drawContours(imgContours,cnt,-1,(255,0,0),3)
            #calculate the perimeter
            peri=cv2.arcLength(cnt,True)
            print('peri',peri)
            #approximation of how many corner plot we have
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            print('approx',len(approx))
            
            objCor=len(approx)
            #give the coordinate of each point of the rectangle 
            #bounding box
            x,y,w,h= cv2.boundingRect(approx)
            
            #-------------------------------------------------      
            #filter for text
            # if 3 corners = triangle
            if objCor ==3: objectType="triangle"

            #if 4 corner = rectangle
            elif objCor ==4:
                aspRatio=w/h
                objectType='rectangle'
                if aspRatio >0.95 and aspRatio <1.05:
                    objectType='square'
                else:
                    objectType='rectangle'
  
            elif objCor > 5:
                objectType='Circle'
            else:
                objectType="None"
            #------------------------------------------------- 
            # draw rectangle+ img+ input coordinate +output coordinate+color+thickness
            cv2.rectangle(imgContours,(x,y),(x+w,y+h),(0,255,0),2)
            #put a text( img+ input coordinate +output coordinate+color+thickness)
            cv2.putText(imgContours,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255),3)      
    return 
#%%
def stackImages_SRC(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]),
                                                None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver
#%%           
def getContours_SRC(img):
    imgContour=img.copy()
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 0), 3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            print(len(approx))
            objCor=len(approx)
            x, y, w, h =cv2.boundingRect(approx)

            if objCor ==3: objectType ="Tri"
            elif objCor == 4:
                aspRatio = w/float(h)
                if aspRatio >1 and aspRatio <1.3: objectType= "Square"
                else:objectType="Rectangle"
            elif objCor>4: objectType= "Circles"
            else:objectType="None"

            cv2.rectangle(imgContour,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.putText(imgContour,objectType,(x+(w//2)-10,y+(h//2)-10),cv2.FONT_HERSHEY_COMPLEX,0.7,
                        (0,0,0),2)
#%%
def findColorBIS(img,Colors):
    """Description:
        prend un dict=> renvois mask d image avec la couleur selectionnée
        la valeur du dict est une liste (encodée en HSV) de 6 éléments tel que:
        lower, upper= np.array([h_min,s_min,v_min]),np.array([h_max,s_max,v_max])
    """ 
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for key in Colors:
        lower, upper= np.array(Colors[key][0:3]),np.array(Colors[key][3:6])
        mask =cv2.inRange(imgHSV,lower,upper)
        cv2.imshow(str(key), mask)

def findColor(img,myColors):
    """Description:
        prend une list=> renvois mask d image avec la couleur selectionnée
         # lower, upper= np.array([h_min,s_min,v_min]),np.array([h_max,s_max,v_max])
    """ 
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower, upper= np.array(color[0:3]),np.array(color[3:6])
        mask =cv2.inRange(imgHSV,lower,upper)
        cv2.imshow(str(color[0]), mask)

def masking(originalImageArray:np.ndarray, minValue:int, maxValue:int):
    if minValue < maxValue:
        return np.clip(originalImageArray, minValue, maxValue, out=None)
    raise Exception(f'minValue < maxValue == {minValue < maxValue}')
