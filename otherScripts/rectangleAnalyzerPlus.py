#%%
# PLot Rectangles
#--------------------------------------------------------------------------------#

# src : https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
# https://programtalk.com/python-examples/skimage.measure.regionprops/

#rectangles
# def polygon(r, c, shape=None):
# def rectangle(start, end=None, extent=None, shape=None):

import math, os 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from skimage.draw import ellipse, rectangle
# rectangle(start, end=None, extent=None, shape=None)
from skimage.measure import label, regionprops, regionprops_table
from skimage import data, filters, measure, morphology
from skimage.transform import rotate

dirpath = os.path.dirname(os.path.realpath(__file__))


image = np.zeros((600, 600))

# polygon_perimeter = rectangle(start=(300, 350), extent=(100,100))
polygon_perimeter = rectangle(start=(300, 350), end=(100, 220))
rr, cc = polygon_perimeter
image[rr, cc] = 1

image = rotate(image, angle=15, order=0)

rr, cc = rectangle(start=(100, 100), end=(60, 50))
image[rr, cc] = 1

label_img = label(image)
regions = regionprops(label_img)

fig, ax = plt.subplots()
ax.imshow(image, cmap=plt.cm.gray)

mask = image
labels = measure.label(mask)
img = mask
#--------------------------------------------------------------------------------#

for props in regions:
    #calcul du points de gravité
    y0, x0 = props.centroid
    
    # compute main axis
    orientation = props.orientation
    x1 = x0 + math.cos(orientation) * 0.5 * props.axis_minor_length
    y1 = y0 - math.sin(orientation) * 0.5 * props.axis_minor_length
    x2 = x0 - math.sin(orientation) * 0.5 * props.axis_major_length
    y2 = y0 - math.cos(orientation) * 0.5 * props.axis_major_length

    # # calcul des axes
    ax.plot((x0, x1), (y0, y1), '-b', linewidth=2.5)
    # ax.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
    ax.plot(x0, y0, 'xg', markersize=15)

    # Definition de la boîte entourant la forme
    minr, minc, maxr, maxc = props.bbox
    bx = (minc, maxc, maxc, minc, minc)
    by = (minr, minr, maxr, maxr, minr)
    ax.plot(bx, by, '-r', linewidth=2.5)

ax.axis((0, 600, 600, 0))
plt.show()
# %
props = regionprops_table(label_img, properties=('centroid',
                                                 'orientation',
                                                 'axis_major_length',
                                                 'axis_minor_length'))
# pd.DataFrame(props)

#--------------------------------------------------------------------------------#

import plotly
import plotly.express as px
import plotly.graph_objects as go

fig = px.imshow(img, binary_string=True)
fig.update_traces(hoverinfo='skip') # hover is only for label info

props = measure.regionprops(labels, img)
properties = ['area', 'perimeter', 'intensity_mean']
# properties = ['area', 'perimeter', 'intensity_mean','coords']

customLabels = ['A69', 'A320']

for index in range(0, labels.max()):
    label_i = props[index].label
    contour = measure.find_contours(labels == label_i, 0.5)[0]
    y0, x0 = regions[index].centroid
    y, x = contour.T
    hoverinfo = ''
    for prop_name in properties:
        hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
    fig.add_trace(
        go.Scatter(
                    x=x, y=y, name=customLabels[index],
                    mode='lines', fill='toself', showlegend=False,
                    hovertemplate=hoverinfo, hoveron='points+fills'
        )
    )
        
    fig.add_trace(
        go.Scatter(
                    x=np.array(x0), y=np.array(y0), name='centerd'
        ),
    )
plotly.io.show(fig)
#%%
# ----------------------------------------------------------
# rectangle
# import PIL
# from PIL import Image, ImageDraw
# w, h = 120, 90
# bbox = [(10, 10), (w - 10, h - 10)]
# img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
# dctx = ImageDraw.Draw(img)  # create drawing context
# dctx.rectangle(bbox, fill="#ddddff", outline="blue")
# del dctx  # destroy drawing context
# img.save(os.path.join(dirpath, "ImageDraw_rectangle_01.jpg"))


#%%
# PLot Rectangles
#--------------------------------------------------------------------------------#

# src : https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_regionprops.html
# https://programtalk.com/python-examples/skimage.measure.regionprops/

#rectangles
# def polygon(r, c, shape=None):
# def rectangle(start, end=None, extent=None, shape=None):


#%%
# import math, os 
# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from skimage.draw import ellipse, rectangle
# # rectangle(start, end=None, extent=None, shape=None)
# from skimage.measure import label, regionprops, regionprops_table
# from skimage import data, filters, measure, morphology
# from skimage.transform import rotate

# dirpath = os.path.dirname(os.path.realpath(__file__))


# image = np.zeros((600, 600))

# # polygon_perimeter = rectangle(start=(300, 350), extent=(100,100))
# polygon_perimeter = rectangle(start=(300, 350), end=(100, 220))
# rr, cc = polygon_perimeter
# image[rr, cc] = 1

# imagerot = rotate(image, angle=15, order=0)
# image = imagerot

# rr, cc = rectangle(start=(100, 100), end=(60, 50))
# image[rr, cc] = 1

# label_img = label(image)
# regions = regionprops(label_img)

# fig, ax = plt.subplots()
# ax.imshow(image, cmap=plt.cm.gray)

# mask = image
# labels = measure.label(mask)
# img = mask
# #--------------------------------------------------------------------------------#

# num = .5
# for props in regions:
#     #calcul du points de gravité
#     y0, x0 = props.centroid
    
    
#     # calcule des coordonnées du carrée détecteur
#     # et de ses dimensions...
#     pstart, pstop = (minx, miny), (maxx, maxy)
#     width, length = abs(pstart[0] - pstop[0]), abs(pstart[1] - pstop[1]) 
#     width2, length2 = math.cos(orientation)*width + math.sin(orientation)*length, math.cos(orientation)*length - math.sin(orientation)*width 
#     # compute main axis
#     # regions[0].orientation*(180/np.pi) red -> deg
#     # regions[1].orientation*(180/np.pi)
#     orientation = props.orientation
#     x1 = x0 + math.cos(orientation) * num * props.axis_minor_length
#     y1 = y0 - math.sin(orientation) * num * props.axis_minor_length
#     x2 = x0 - math.sin(orientation) * num * props.axis_major_length
#     y2 = y0 - math.cos(orientation) * num * props.axis_major_length

#     # # calcul des axes
#     ax.plot((x0, x1), (y0, y1), '-b', linewidth=2.5)
#     ax.plot((x0, x2), (y0, y2), '-b', linewidth=2.5)
#     ax.plot(x0, y0, 'xg', markersize=15)

#     # Definition de la boîte entourant la forme
#     minx, miny, maxx, maxy = props.bbox
#     bx = (minx, maxx, maxx, minx, minx)
#     by = (miny, miny, maxy, maxy, miny)
#     ax.plot(bx, by, '-r', linewidth=2.5)

# ax.axis((0, 600, 600, 0))
# plt.show()
# # %
# props = regionprops_table(label_img, properties=('centroid',
#                                                  'orientation',
#                                                  'axis_major_length',
#                                                  'axis_minor_length'))
# # pd.DataFrame(props)
