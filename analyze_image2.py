#!/usr/bin/env python

######################
#   Matplotlib based image analyzer
#   Version 2.0
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
    # at luma = 0.5, Pr = 0.0, Pb ranges from RGB(127, 171, 0) to RGB(127.5, 83.62, 255)
    
    cdict_pb = {
        'red': (( 0.0, 0.0, 127.0/255),
                (1.0, 127.0/255, 127.0/255)),
    
        'green': (( 0.0, 0.0, 83.6/255),
                  ( 1.0, 171.0/255, 171.0/255)),
        
        'blue':  (( 0.0, 0.0, 255.0/255),
                  ( 1.0, 0.0, 0.0/255))
        }
    # at luma = 0.5, Pb = 0.0, Pr ranges from RGB(0, 218, 127) to RGB(255, 36, 127)
    cdict_pr =  {'red':   ((0.0, 0.0, 255.0/255),
                           (1.0, 0.0/255, 0.0/255)),

         'green': (( 0.0, 0.0, 36.0/255),
                   ( 1.0, 218.0/255, 218.0/255)),

         'blue':  (( 0.0, 0.0, 127.0/255),
                   ( 1.0, 127.0/255, 127.0/255))
        }

    
    colormaps["red"] = LinearSegmentedColormap('testcmap', cdict_red)
    colormaps["grn"] = LinearSegmentedColormap('testcmap', cdict_grn)
    colormaps["blu"] = LinearSegmentedColormap('testcmap', cdict_blu)
    colormaps["pb"] = LinearSegmentedColormap('testcmap', cdict_pb)
    colormaps["pr"] = LinearSegmentedColormap('testcmap', cdict_pr)
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

plotluma = plots[1][0].imshow(luma, cmap="Greys_r", vmin=0, vmax=255)
plots[1][0].set_title("Luma (Y)")
fig.colorbar(plotluma, ax=plots[1][0])

plotcb = plots[1][1].imshow(pb, cmap=colormaps["pb"], vmin=-127, vmax=127)
plots[1][1].set_title("Blue difference (Pb)")
fig.colorbar(plotcb, ax=plots[1][1])

plotcr = plots[1][2].imshow(pr, cmap=colormaps["pr"], vmin=-127, vmax=127)
plots[1][2].set_title("Red difference (Cr)")
fig.colorbar(plotcr, ax=plots[1][2])

plt.show()

#fig.savefig(impath+"-output.png")