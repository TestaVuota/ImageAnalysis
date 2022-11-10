#%%

# https://hhsprings.bitbucket.io/docs/programming/examples/python/PIL/ImageDraw.html

import math, sys, os
from PIL import Image, ImageDraw
dirpath = os.path.dirname(os.path.realpath(__file__))
try:
    os.makedirs(os.path.join(os.path.dirname(dirpath), "result"))
except:
    pass
#
w, h = 120, 90
bbox = [(10, 10), (w - 10, h - 10)]

# ----------------------------------------------------------
# rectangle
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.rectangle(bbox, fill="#ddddff", outline="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_rectangle_01.jpg"))

# ----------------------------------------------------------
# ellipse
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.ellipse(bbox, fill="#ddddff", outline="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_ellipse_01.jpg"))

# ----------------------------------------------------------
# arc (end=130)
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.arc(bbox, start=20, end=130, fill="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_arc_01.jpg"))

# chord (end=130)
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.chord(bbox, start=20, end=130, fill="#ddddff", outline="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_chord_01.jpg"))

# pieslice (end=130)
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.pieslice(bbox, start=20, end=130, fill="#ddddff", outline="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_pieslice_01.jpg"))

# ----------------------------------------------------------
# arc (end=300)
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.arc(bbox, start=20, end=300, fill="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_arc_02.jpg"))

# chord (end=300)
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.chord(bbox, start=20, end=300, fill="#ddddff", outline="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_chord_02.jpg"))

# pieslice (end=300)
img = Image.new("RGB", (w, h), "#f9f9f9")  # create new Image
dctx = ImageDraw.Draw(img)  # create drawing context
dctx.pieslice(bbox, start=20, end=300, fill="#ddddff", outline="blue")
del dctx  # destroy drawing context
img.save(os.path.join(dirpath, "result\\ImageDraw_pieslice_02.jpg"))
#%%