#%%
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
##arrayLucc is the array of land use types 
arrayLucc = np.random.randint(1,4,(5,5))
## first you need to define your color map and value name as a dic
t = 1 ## alpha value
cmap = {1:[0.1,0.1,1.0,t],2:[1.0,0.1,0.1,t],3:[1.0,0.5,0.1,t]}
labels = {1:'agricultural land',2:'forest land',3:'grassland'}
arrayShow = np.array([[cmap[i] for i in j] for j in arrayLucc])    
## create patches as legend
patches =[mpatches.Patch(color=cmap[i],label=labels[i]) for i in cmap]

plt.imshow(arrayShow)
plt.legend(handles=patches, loc=4, borderaxespad=0.)
plt.show()
#%%