#!/usr/bin/env python

######################
#   Matplotlib based image analyzer
#   polprog 2018
#   3 clause BSD licensed
######################



import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import matplotlib.gridspec as gspc
from matplotlib.colors import LinearSegmentedColormap


COEF_R = 0.2126
COEF_G = 0.7152
COEF_B = 0.0722



def setup_colormaps():
    colormaps = {}
    cdict_red = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
    cdict_grn = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0))
        }
    cdict_blu = {'red':   ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'green': ((0.0, 0.0, 0.0),
                   (1.0, 0.0, 0.0)),

         'blue':  ((0.0, 0.0, 0.0),
                   (1.0, 1.0, 1.0))
        }
    colormaps["red"] = LinearSegmentedColormap('testcmap', cdict_red)
    colormaps["grn"] = LinearSegmentedColormap('testcmap', cdict_grn)
    colormaps["blu"] = LinearSegmentedColormap('testcmap', cdict_blu)   
    return  colormaps





print("Enter image path")
impath=input()

print("Loading ", impath)



imag = mpimg.imread(impath)


red = imag[:, :, 0]
grn = imag[:, :, 1]
blu = imag[:, :, 2]

luma = imag[:, :, 0] * COEF_R + imag[:, :, 1] * COEF_G + imag[:, :, 2] * COEF_B

pb = luma - blu
pr = luma - red

colormaps = setup_colormaps()
print(colormaps)


fig, plots = plt.subplots(2, 3, squeeze=True,
                          gridspec_kw={"hspace": 0.5, "wspace":0.7},
                          constrained_layout=True,
                          figsize= (20, 20),)



plotred = plots[0][0].imshow(red, cmap=colormaps["red"])
plots[0][0].set_title("Red")
fig.colorbar(plotred, ax=plots[0][0])

plotgrn = plots[0][1].imshow(grn, cmap=colormaps["grn"])
plots[0][1].set_title("Green")
fig.colorbar(plotgrn, ax=plots[0][1])

plotblu = plots[0][2].imshow(blu, cmap=colormaps["blu"])
plots[0][2].set_title("Blue")
fig.colorbar(plotblu, ax=plots[0][2])

plotluma = plots[1][0].imshow(luma, cmap="Greys_r")
plots[1][0].set_title("Luma")
fig.colorbar(plotluma, ax=plots[1][0])

plotcb = plots[1][1].imshow(pb, cmap="Greys_r")
plots[1][1].set_title("Cb")
fig.colorbar(plotcb, ax=plots[1][1])

plotcr = plots[1][2].imshow(pr, cmap="Greys_r")
plots[1][2].set_title("Cr")
fig.colorbar(plotcb, ax=plots[1][2])

plt.show()

#fig.savefig(impath+"-output.png")