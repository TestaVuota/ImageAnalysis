# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 15:45:27 2022

@author: lilulu


This method used to implement the slider method. In the 690*690 pixels within the glue,
a 30*30 area is selected for the slider fit. Find the best-fitted area.
"""

# https://tel.archives-ouvertes.fr/tel-02191661/file/78868_MARIN_2019_archivage.pdf

import numpy as np
import matplotlib as plt
import cv2
from scipy.optimize import nnls
from PIL import Image
from pylab import *
import pandas as pd

path = 'F:\\3.5\\programs\\data SPARC\\complicated_database\\40\\'
path = r'C:\Users\nicol\OneDrive\Bureau\lulu\PortII'
# path = 'F:\\3.5\\programs\\data SPARC\\data base\\PORTI\\30\\'
path_test = 'F:\\3.5\\programs\\data SPARC\\signal test\\Mesures_multispectrales\\'
path_test_c = 'F:\\3.5\\programs\\data SPARC\\signal test\\Mesures_multispectrales\\complicated_database\\'

test1 = path_test +'PORTI\\SP1\\S1P1.tiff'
test2 = path_test +'PORTI\\SP2\\S2P1.tiff'
test3 = path_test +'PORTI\\SP3\\S3P1.tiff'
test4 = path_test +'PORTI\\SP4\\S4P1.tiff'

test5 = path_test +'PORTII\\SP1\\S1P2.tiff'
test6 = path_test +'PORTII\\SP2\\S2P2.tiff'
test7 = path_test +'PORTII\\SP3\\S3P2.tiff'
test8 = path_test +'PORTII\\SP4\\S4P2.tiff'

# s1p1, s2p2, s3p3, s4p4, s1p2, s2p2, s3p2, s4p2
s1 = path_test +'PORTI\\SP1\\SAV SPU.TXT'
s2 = path_test +'PORTI\\SP2\\SAV SPD.TXT'
s3 = path_test +'PORTI\\SP3\\SAV SPT.TXT'
s4 = path_test +'PORTI\\SP4\\SAV SPQ.TXT'

s5 = path_test +'PORTII\\SP1\\SAV SPU.TXT'
s6 = path_test +'PORTII\\SP2\\SAV SPD.TXT'
s7 = path_test +'PORTII\\SP3\\SAV SPT.TXT'
s8 = path_test +'PORTII\\SP4\\SAV SPQ.TXT'

Z = {
     '1':(672, 439),
     '2':(703, 439),
     '3':(734, 439),
     '4':(672, 470),
     '5':(672, 501),
     '6':(672, 532),
     '7':(703, 740),
     '8':(703, 501),
     '9':(734, 470),
     '10':(734, 501),
     '11':(641, 470),
     '12':(641, 501),
     '13':(610, 470),
     '14':(610, 501),
     '15':(610, 532),
     '16':(641, 532),
     '17':(703, 532),
     '18':(734, 532),
     '19':(641, 439),
     '20':(610, 439),
     '21':(363, 144),  # square area 690 by 690 piexls inside glue
     '22':(594, 410)  # Brighter areas 69 by 34
     }
slider_len = 30
num = 690/slider_len
merge_areas = 0
area = 21       # different areas here
imgs_y, imgs_x = Z[str(area)]
index = 0
for num_i in range(0, int(num)):
    for num_j in range(0, int(num)):
        imgs = [] # all images
        imgs_Y, imgs_X = imgs_y + slider_len *num_j, imgs_x + slider_len * num_i
        
        for i in range(750,851):
            img = cv2.imread(''.join([path,"\\"+str(i),".tiff"]), cv2.IMREAD_GRAYSCALE)
            #img = array(Image.open(path + str(i) + '.tiff'))
            img = img[imgs_X:imgs_X+slider_len, imgs_Y:imgs_Y+slider_len] #area z4
            imgs.append(reshape(img, slider_len*slider_len))
        #imgs = np.array(imgs).astype('uint16')
        imgs = np.array(imgs)
        # merge_areas = merge_areas + np.double(imgs)/5
        # imgs = np.uint8(merge_areas)
        #imgs.shape
        
         
        #read test tiff img and right txt spectrum
        # test 1 2
        X = [] # Predictive Spectrum
        Y = [] # Right spectrum
        for i in range(1,9):
            # # test s1p1...s1p2...
            # #R = cv2.imread('test' + str(i))
            # R = cv2.imread(eval("test"+str(i)), cv2.IMREAD_GRAYSCALE)
            # R = R[imgs_X:imgs_X+21, imgs_Y:imgs_Y+21]
            # #print(R.shape)
            # R = np.array(R.reshape(441))
            # x, rnorm = nnls(imgs.T, R) # NNLS
            # X.append(x)
            # S1 S2 S3 S4
            S = pd.read_csv(eval('s'+str(i)),skiprows=23,delimiter=',').to_numpy()
            y = S[:,1]  # The right spectrum
            Y.append(y)
           
        
        # test complicated data
        for i in range(1, 5):
            R = cv2.imread(path_test_c + str(i) + '.tiff', cv2.IMREAD_GRAYSCALE)
            R = R[imgs_X:imgs_X+slider_len, imgs_Y:imgs_Y+slider_len]
            R = np.array(R.reshape(slider_len*slider_len))
            x, rnorm = nnls(imgs.T, R) # NNLS
            X.append(x)
        
        # Plot
        fig,axs = plt.subplots(2, 2)
        axs[0,0].plot(range(750,851), X[0]/np.abs((max(X[0])-min(X[0]))))
        axs[0,0].plot(range(750,851), Y[0]/np.abs((max(Y[0])-min(Y[0]))))
        axs[0,0].set_title("S1")
        axs[0,1].plot(range(750,851), X[1]/np.abs((max(X[1])-min(X[1]))))
        axs[0,1].plot(range(750,851), Y[1]/np.abs((max(Y[1])-min(Y[1]))))
        axs[0,1].set_title("S2")
        axs[1,0].plot(range(750,851), X[2]/np.abs((max(X[2])-min(X[2]))))
        axs[1,0].plot(range(750,851), Y[2]/np.abs((max(Y[2])-min(Y[2]))))
        axs[1,0].set_title("S3")
        axs[1,1].plot(range(750,851), X[3]/np.abs((max(X[3])-min(X[3]))), label = 'slider:' + str(num_i)+','+ str(num_j))
        axs[1,1].plot(range(750,851), Y[3]/np.abs((max(Y[3])-min(Y[3]))), label = 'norm')
        axs[1,1].set_title("S4")
        fig.legend(loc = 3, bbox_to_anchor=(1, 0), borderaxespad=0)
        fig.text(0.5, 0, 'wavelength', ha = 'center')
        fig.text(0, 0.5, 'Normalized intensity', va = 'center', rotation = 'vertical')
        plt.suptitle('use  '+ path[-24:-1]+'-areas'+ str(area) +'-slider:' + str(num_i)+','+ str(num_j)+'-NNLS')
        plt.tight_layout()
        
        plt.savefig('F:\\3.5\\训练图\\slider_\\' + str(index) + '.png')
        index = index +1
        plt.show()
         
        
        # fig,axs = plt.subplots(2, 2)
        # axs[0,0].plot(range(750,851), X[4]/np.abs((max(X[4])-min(X[4]))))
        # axs[0,0].plot(range(750,851), Y[4]/np.abs((max(Y[4])-min(Y[4]))))
        # axs[0,0].set_title("S1")
        # axs[0,1].plot(range(750,851), X[5]/np.abs((max(X[5])-min(X[5]))))
        # axs[0,1].plot(range(750,851), Y[5]/np.abs((max(Y[5])-min(Y[5]))))
        # axs[0,1].set_title("S2")
        # axs[1,0].plot(range(750,851), X[6]/np.abs((max(X[6])-min(X[6]))))
        # axs[1,0].plot(range(750,851), Y[6]/np.abs((max(Y[6])-min(Y[6]))))
        # axs[1,0].set_title("S3")
        # axs[1,1].plot(range(750,851), X[7]/np.abs((max(X[7])-min(X[7]))), label = 'z'+str(area))
        # axs[1,1].plot(range(750,851), Y[7]/np.abs((max(Y[7])-min(Y[7]))), label = 'P2')
        # axs[1,1].set_title("S4")
        # fig.legend(loc = 3, bbox_to_anchor=(1, 0), borderaxespad=0)
        # fig.text(0.5, 0, 'wavelength', ha = 'center')
        # fig.text(0, 0.5, 'Normalized intensity', va = 'center', rotation = 'vertical')
        # plt.suptitle('use   '+path[-24:-1]+' areas'+ str(area) +'slider:' + str(num_i)+','+ str(num_j)+'-NNLS')
        # plt.tight_layout()
    
'''
# plot
fig,axs = plt.subplots(4, 2)
axs[0,0].plot(range(750,851), X[0]/np.abs((max(X[0])-min(X[0]))))
axs[0,0].set_title("S1P1")
axs[0,1].plot(range(750,851), X[1]/np.abs((max(X[1])-min(X[1]))))
axs[0,1].set_title("S2P1")
axs[1,0].plot(range(750,851), X[2]/np.abs((max(X[2])-min(X[2]))))
axs[1,0].set_title("S3P1")
axs[1,1].plot(range(750,851), X[3]/np.abs((max(X[3])-min(X[3]))))
axs[1,1].set_title("S4P2")
axs[2,0].plot(range(750,851), X[4]/np.abs((max(X[4])-min(X[4]))))
axs[2,0].set_title("S1P2")
axs[2,1].plot(range(750,851), X[5]/np.abs((max(X[5])-min(X[5]))))
axs[2,1].set_title("S2P2")
axs[3,0].plot(range(750,851), X[6]/np.abs((max(X[6])-min(X[6]))))
axs[3,0].set_title("S3P2")
axs[3,1].plot(range(750,851), X[7]/np.abs((max(X[7])-min(X[7]))))
axs[3,1].set_title("S4P2")
plt.tight_layout()

fig,axs = plt.subplots(2, 4)
axs[0,0].plot(range(750,851), X[0]/np.abs((max(X[0])-min(X[0]))))
axs[0,0].set_title("S1P1")
axs[0,1].plot(range(750,851), X[1]/np.abs((max(X[1])-min(X[1]))))
axs[0,1].set_title("S2P1")
axs[0,2].plot(range(750,851), X[2]/np.abs((max(X[2])-min(X[2]))))
axs[0,2].set_title("S3P1")
axs[0,3].plot(range(750,851), X[3]/np.abs((max(X[3])-min(X[3]))))
axs[0,3].set_title("S4P2")
axs[1,0].plot(range(750,851), X[4]/np.abs((max(X[4])-min(X[4]))))
axs[1,0].set_title("S1P2")
axs[1,1].plot(range(750,851), X[5]/np.abs((max(X[5])-min(X[5]))))
axs[1,1].set_title("S2P2")
axs[1,2].plot(range(750,851), X[6]/np.abs((max(X[6])-min(X[6]))))
axs[1,2].set_title("S3P2")
axs[1,3].plot(range(750,851), X[7]/np.abs((max(X[7])-min(X[7]))))
axs[1,3].set_title("S4P2")
plt.tight_layout()
'''


'''# test z4 area location
# img = array(Image.open(path + '750.tiff').convert('GRAY'))
# imshow(img)
# lu = [672, 693]
# rb = [470, 470]    
# axis('off')
# plot(img[lu, rb], 'go-')
# plt.scatter(672, 470, s = 3)
# plt.scatter(693, 470, s = 3)
# plt.scatter(672, 491, s = 3)
# plt.scatter(693, 491, s = 3)

'''

