from pylab import *

from mnist import MNIST    # import the mnist class

mndata = MNIST('./data')    # init with the 'data' dir

# load data
mndata.load_training() 
mndata.load_testing()

# the number of training patterns
training_length = len(mndata.train_images)

# the number of test patterns
test_length = len(mndata.test_images)

# the number of pixels per side of all images
img_side = 28

# each input is a raw vector.
# the number of units of the network 
# corresponds to the number of input elements
n_pixels = img_side*img_side 

#-------------------------------------------------------------
# a custom plot that uses imshow to draw a matrix
# x:        array           the matrix to be plotted
# fig:      figure object   figure device to use
# window:   int             the current subplot position
# windows   int             number of subplot
def plot_img(x, fig, window, windows = 8) :
    ax = fig.add_subplot(1, windows, window)
    ax.imshow(x, interpolation = 'none', 
              aspect = 'auto', cmap=cm.Greys)  
    axis('off')
    fig.canvas.draw()

#-------------------------------------------------------------
# transform a raw input in an image matrix  
# x:      array    the raw input vector
# return  array    a squared matrix
def to_mat(x) :
    return x.reshape( img_side, img_side )

#-------------------------------------------------------------
# step function
# return:    1 if x > 0
#            0 otherwise
def step(x) :
    return 1.0*(x>0)

#-------------------------------------------------------------
# sigmoid function
# t   float temperature
def sigmoid(x,t=1.0) :
    return 1.0/(1.0 + exp(-x/t))

#-------------------------------------------------------------
# RELU: rectifier linear unit function
# t   float temperature
def sigmoid(x) :
    return maximum(0,x)

#-------------------------------------------------------------
# Create an array with 2-dimensional patterns belonging to two categories
# n_patterns    :        int                 Number of patterns
# std_deviation :        float               Standard deviation of noise
# centroids1    :        2-elements-vectors  Points from which the
#                        list                patterns belonging to 
#                                            the class are generated
# centroids2    :        2-elements-vectors  Points from which the
#                        list                patterns not belonging 
#                                            to the class are generated
# returns       :        array               Each row contains a pattern as
#                                            its first two elements and
#                                            the group (belonging/not 
#                                            belonging/) as its third element
def build_dataset( n_patterns = 100, 
                   std_deviation = 0.2,
                   centroids1 = [ [-1.2, 1.8], [-1.8, 1.2] ], 
                   centroids2 = [ [-0.2, 0.4], [-0.8, -0.2 ] ] ) :
    
    # Decide to which group patterns are from.
    # First half belongs to the class
    categories = arange(n_patterns)/(n_patterns/2)   
    
    # Each row of this array will contain a 2-element-wide 
    # input pattern plus an integer defining to which category
    # the pattern belongs   
    data = zeros([n_patterns,3])

    # Iterate the patterns to generate
    for t in xrange(n_patterns) :
        
        pattern = zeros(2)
        
        if categories[t] > 0 : 
            index = int( rand()*len(centroids1) )
            pattern = array(centroids1[index])
        else :
            index = int( rand()*len(centroids2) )
            pattern = array(centroids2[index])
        
        # Add noise to each element of the centroid
        pattern += std_deviation*randn(2)

        # Fill up data 
        data[t,:] = hstack([pattern, categories[t]])
    
    return data




