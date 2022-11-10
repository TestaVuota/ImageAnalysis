# -*- coding: utf-8 -*-
"""
Created on Wed Sep 21 18:48:35 2022

@author: nicol
"""

# src : https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html
# src : https://fr.acervolima.com/python-pil-imagedraw-draw-rectangle/
# importing image object from PIL
import sys, math
from PIL import Image, ImageDraw

imgpath = r'C:\Users\nicol\Documents\Python\Detection_rectangle\700.0p5e-06.tiff'
with Image.open(imgpath) as im:

    draw = ImageDraw.Draw(im)
    draw.line((0, 0) + im.size, fill=128)
    draw.line((0, im.size[1], im.size[0], 0), fill=128)

    
    w, h = 220, 190
    # w, h = im.size
    shape = [(40, 40), (w - 10, h - 10)]
    draw.rectangle(shape, outline ="red")
     

    im.show()
    
#%%

# Regarder si existe fonction pour draw polygone à partir d un set de points
# src : https://pillow.readthedocs.io/en/stable/reference/ImageDraw.html

#
# Draw polygon with PIL
#

import math
from PIL import Image, ImageDraw
from PIL import ImagePath 
# src : https://www.geeksforgeeks.org/python-pil-imagedraw-draw-polygon-method/  
side = 8
xy = [
    ((math.cos(th) + 1) * 90,
     (math.sin(th) + 1) * 60)
    for th in [i * (2 * math.pi) / side for i in range(side)]
    ]  
# obtenir la taille de l'image?
image = ImagePath.Path(xy).getbbox()  
# math ceil -> https://www.geeksforgeeks.org/python-math-ceil-function/
size = list(map(int, map(math.ceil, image[2:])))
  
img = Image.new("RGB", size) 
img1 = ImageDraw.Draw(img)  
img1.polygon(xy, outline ="blue") 
  

print('xy : ',xy,'image : ',image,'size : ',size)

img.show()

#%%
# https://pillow.readthedocs.io/en/stable/handbook/tutorial.html?highlight=Image.new#merging-images
def merge(im1, im2):
    w = im1.size[0] + im2.size[0]
    h = max(im1.size[1], im2.size[1])
    im = Image.new("RGBA", (w, h))

    im.paste(im1)
    im.paste(im2, (im1.size[0], 0))

    return im