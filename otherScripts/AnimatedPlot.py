# https://matplotlib.org/stable/gallery/animation/simple_anim.html
# https://stackoverflow.com/questions/25482876/how-to-add-legend-to-imshow-in-matplotlib
#%% 
A = {
        '01': (242, 233), '02': (274, 233), '03': (305, 233), '04': (336, 233), '05': (367, 233),
        '11': (242, 262), '12': (274, 262), '13': (305, 262), '14': (336, 262), '15': (367, 262),
        '21': (242, 294), '22': (274, 294), '23': (305, 294), '24': (336, 294), '25': (367, 294),
        '31': (242, 324), '32': (274, 324), '33': (305, 324), '34': (336, 324), '35': (367, 324),
        '41': (242, 354), '42': (274, 354), '43': (305, 354), '44': (336, 354), '45': (367, 354),
        '51': (242, 383), '52': (274, 383), '53': (305, 383), '54': (336, 383), '55': (367, 383),
        '61': (242, 412), '62': (274, 412), '63': (305, 412), '64': (336, 412), '65': (367, 412),
    }

from utils import dirpath, locateFileExt, locateFileNameExt
import numpy as np
import os

basepath_ = os.path.join(dirpath(),'canaux')
basepaths = locateFileNameExt(path=basepath_, containInName="stackImagePower5.0um", ext=".npy")
basepaths.sort() 
print([basepath.split('\\')[-1].split('.')[1] for basepath in basepaths])

#loading images
stackImages = np.load(basepaths[0])
Aimages = {f"{list(A.keys())[i]}":image for i, image in enumerate(stackImages)}
img = Aimages['43']
nlbd = img.shape[0]
x = y = int(np.sqrt(img.shape[1]))
img_ = img.reshape(nlbd,x,y)
#----------------------------------------------------------------------------------------------------------------------------#
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.pylab as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

fig, ax = plt.subplots()
lbd = np.linspace(700,850,img.shape[0])

# divider = make_axes_locatable(ax)
# cax = divider.append_axes("right", size="5%", pad=0.05)
# fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax)

ims, num_steps = [], 30

for i in range(img_.shape[0]):
    titlestr =  f'{round(lbd[i],1)} nm'
    text = ax.text(20, 25, titlestr, bbox={'facecolor': 'white', 'pad': 2})
    
    im = ax.imshow(img_[i,:,:], cmap='gray', animated=True, label=titlestr)
    if i == 0:
        ax.imshow(img_[0,:,:], cmap='gray')  # show an initial one first
    ims.append([im,text])

ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True,
                                repeat_delay=1000)
# ani = animation.FuncAnimation(
#     fig, img, num_steps, interval=100)
fig.colorbar(im)
plt.legend()
plt.show()

# src : https://stackoverflow.com/questions/62644614/requested-moviewriter-ffmpeg-not-available-even-when-i-installed-it
# src : https://stackoverflow.com/questions/58469357/python-unknown-file-extension-mp4
# src : https://holypython.com/how-to-save-matplotlib-animations-the-ultimate-guide/
# src : https://stackoverflow.com/questions/10360327/file-format-for-video-analysis/


# fps_ = (5, 10, 20, 30)
# for fps in fps_[:]:
#     savepath = os.path.join(basepath_,f'filename{fps}fps.gif')
#     writergif = animation.PillowWriter(fps=fps)
#     ani.save(savepath,writer=writergif)

# #Format Video
# https://matplotlib.org/stable/api/animation_api.html

#----------------------------------------------------------------------------------------------------------------------------#
# https://www.geeksforgeeks.org/moviepy-creating-animation-using-matplotlib/    

# # importing movie py libraries
# from moviepy.editor import VideoClip
# from moviepy.video.io.bindings import mplfig_to_npimage

# # duration of the video
# duration = 2

# # creating animation
# animation = VideoClip(ims, duration = duration)

# # displaying animation with auto play and looping
# animation.ipython_display(fps = 20, loop = True, autoplay = True)

#----------------------------------------------------------------------------------------------------------------------------#
formats_ = ['mp4','avi','mov',]
fps = 10

savepath = os.path.join(basepath_,f'filename{fps}fps.{formats_[0]}')
# moviewriter = animation.MovieWriter
# moviewriter = moviewriter.setup(fig, savepath)
with animation.MovieWriter.saving(fig, savepath, dpi=100):
    for j in range(nlbd):
        update_figure(j)
        moviewriter.grab_frame()
moviewriter.finish()
#----------------------------------------------------------------------------------------------------------------------------#

# formats_ = ['mp4','avi','mov',]
# savepath = os.path.join(basepath_,f'filename{fps}fps.{formats_[0]}')
# writervideo = animation.FFMpegWriter(fps=fps) 
# ani.save(savepath,writer=writervideo)

# for format_ in formats_:
#     savepath = os.path.join(basepath_,f'filename{fps}fps.{format_}')
#     writervideo = animation.FFMpegWriter(fps=fps) 
#     ani.save(savepath,writer=writervideo)

# import matplotlib as mpl 
# mpl.rcParams['animation.ffmpeg_path'] = r'C:\\Users\\xx\\Desktop\\ffmpeg\\bin\\ffmpeg.exe'

#%%
import imageio
savepath = os.path.join(basepath_,f'filename.gif')
gif = imageio.get_reader(savepath)

# Here's the number you're looking for
number_of_frames = len(gif)

frames = []
for idx, frame in enumerate(gif):
    frames.append(frame)
# each frame is a numpy matrix
frames = np.array(frames)
print(frames.shape)

#%%





































#%%
# import PIL 
# from PIL import Image
# savepath = os.path.join(basepath_,f'filename.gif')
# im = Image.open(savepath)
# print(im.size)

# # To iterate through the entire gif
# try:
#     while 1:
#         im.seek(im.tell()+1)
#         # do something to im
# except EOFError:
#     pass # end of sequence
# #%%
# class GIFError(Exception): pass

# def get_gif_num_frames(filename):
#     frames = 0
#     with open(filename, 'rb') as f:
#         if f.read(6) not in ('GIF87a', 'GIF89a'):
#             raise GIFError('not a valid GIF file')
#         f.seek(4, 1)
#         def skip_color_table(flags):
#             if flags & 0x80: f.seek(3 << ((flags & 7) + 1), 1)
#         flags = ord(f.read(1))
#         f.seek(2, 1)
#         skip_color_table(flags)
#         while True:
#             block = f.read(1)
#             if block == ';': break
#             if block == '!': f.seek(1, 1)
#             elif block == ',':
#                 frames += 1
#                 f.seek(8, 1)
#                 skip_color_table(ord(f.read(1)))
#                 f.seek(1, 1)
#             else: raise GIFError('unknown block type')
#             while True:
#                 l = ord(f.read(1))
#                 if not l: break
#                 f.seek(l, 1)
#     return frames
# frames = get_gif_num_frames(savepath)
# https://stackoverflow.com/questions/10360327/file-format-for-video-analysis
#%%
# """
# =================================
# Pausing and Resuming an Animation
# =================================
# """

# class PauseAnimation:
#     def __init__(self, image):
#         fig, ax = plt.subplots()
#         lbd = np.linspace(700,850,img_.shape[0])

#         # Start to stack artist in artists
#         self.ims = []

#         for i in range(img_.shape[0]):
#             titlestr =  f'{round(lbd[i],1)} nm'
#             text = ax.text(20, 25, titlestr, bbox={'facecolor': 'white', 'pad': 2})
            
#             im = ax.imshow(img_[i,:,:], animated=True, label=titlestr)
#             if i == 0:
#                 ax.imshow(img_[0,:,:])  # show an initial one first

#             self.ims.append([im,text])

#         # Defining the animation
#         self.animation = animation.ArtistAnimation(fig, self.ims, interval=50, blit=True,
#                                 repeat_delay=1000)

#         # Defining the event detector
#         self.paused = False
#         fig.canvas.mpl_connect('button_press_event', self.toggle_pause)

#     def toggle_pause(self, *args, **kwargs):
#         if self.paused:
#             self.animation.resume()
#         else:
#             self.animation.pause()
#         self.paused = not self.paused

# if __name__ == '__main__':
#     pa = PauseAnimation(img_)
#     plt.legend()
#     plt.show()

# #%%
# # Background cancellation 
# # https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_rolling_ball.html#sphx-glr-auto-examples-segmentation-plot-rolling-ball-py

