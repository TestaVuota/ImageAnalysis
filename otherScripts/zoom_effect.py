#%%
# src : https://towardsdatascience.com/zooming-in-and-zooming-out-in-matplotlib-to-better-understand-the-data-b4a5f5b4107d
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = r'C:\Users\nicol\Downloads\archive (1)'
filename = "auto_clean.csv"
d = pd.read_csv(os.path.join(path,filename))
print(d)
#%%
fig = plt.figure(figsize = (8, 6))
x = d['length']
y = d['width']
c = d['price']
ax = plt.scatter(x, y, s = 25, c = c)
plt.xlabel('Length', labelpad = 8)
plt.ylabel('Width', labelpad = 8)
plt.title("Length vs Width and Color Represents the Changes of Price")
ax_new = fig.add_axes([0.2, 0.7, 0.2, 0.2])
plt.scatter(x, y, s=5, c = c)



from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
plt.figure(figsize = (8, 5))
x = d['length']
y = d['price']
ax = plt.subplot(1, 1, 1)
ax.scatter(x, y)
ax.set_xlabel("Length")
ax.set_ylabel("Price")
#Defines the size of the zoom window and the positioning
axins = inset_axes(ax, 1, 1, loc = 1, bbox_to_anchor=(0.3, 0.7),
                   bbox_transform = ax.figure.transFigure)
axins.scatter(x, y)
x1, x2 = 0.822, 0.838
y1, y2 = 6400, 12000
#Setting the limit of x and y direction to define which portion to #zoom
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)
#Draw the lines from the portion to zoom and the zoom window
mark_inset(ax, axins, loc1=1, loc2=3, fc="none", ec = "0.4")
plt.show()
#%%