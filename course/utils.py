from pylab import *

from mnist import MNIST    # import the mnist class

mndata = MNIST('./data')    # init with the 'data' dir

# load data
mndata.load_training() 
mndata.load_testing()

# the number of pixels per side of all images
img_side = 28

# each input is a raw vector.
# the number of units of the network 
# corresponds to the number of input elements
n = img_side*img_side 

# set the maximum number of plots to be printed in a row
windows = 8

# a custom plot that uses imshow to draw a matrix
# x:        array           the matrix to be plotted
# fig:      figure object   figure device to use
# window:   int             the current subplot position
# windows   int             number of subplot
def plot_img(x, fig, window, windows = windows) :
    ax = fig.add_subplot(1, windows, window)
    ax.imshow(x, interpolation = 'none', 
              aspect = 'auto', cmap=cm.Greys)  
    axis('off')
    fig.canvas.draw()


# transform a raw input in an image matrix  
# x:      array    the raw input vector
# return  array    a squared matrix
def to_mat(x) :
    return x.reshape( img_side, img_side )
