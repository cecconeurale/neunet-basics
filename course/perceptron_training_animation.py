# The matplotlib object to do animations
from pylab import *
from utils import *
from matplotlib import animation

# This grid allows to layout subplots in a more
# flexible way
import matplotlib.gridspec as gridspec


class AnimatePerceptron :

	
    def __init__(self, input_store, output_store, label_store):

        self.input_store =  input_store
        self.output_store = output_store
        self.label_store = label_store

        self.m, self.trials = self.output_store.shape


    # Initialize the figure for the animation    
    # target_index :  int     The index of the target 
    #                         to plot
    # returns      :  tuple   The three plotting objects 
    #                         to render       
    def init_fig(self, timestep = 0, plot_error = True) :    

        # This is the input digit
        input_digit = self.input_store[:,timestep]

        # This is the output of the network 
        # (10-elements vector)
        output = self.output_store[:,timestep]

        # This is the target of the network 
        # (10-elements vector)
        target = self.label_store[:,timestep]


        m = self.output_store.shape[0]
        trials = self.output_store.shape[1]

        # Init the grid and the figure
        gs = gridspec.GridSpec(8, 24)
        self.fig = figure(figsize=(10, 4.5))        




        #-------------------------------------------------
        # Plot 1 - plot the input digit

        # Create subplot
        ax1 = self.fig.add_subplot(gs[:4,:4])

        title("input")

        # Create the imshow and save the handler
        self.im_input = ax1.imshow(to_mat(input_digit), 
                interpolation = 'none', 
                aspect = 'auto',
                cmap = cm.binary) 
        # Further plot specs
        axis('off')




        #-------------------------------------------------
        # Plot 2 - plot the current state of the network

        # Create subplot
        ax2 = self.fig.add_subplot(gs[:4,6:])

        title("output vector")

        # Create the imshow and save the handler
         
        self.im_output = ax2.bar(arange(m), output, 
                width=.7, color="blue", align="center") 
        self.im_target = ax2.bar(arange(m)-.5, target, 
                width=.3, color="red", align="center") 

        # Only bottom axes
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.set_xticks(arange(m)+.5)
        ax2.set_xticklabels(arange(m))
        ax2.set_yticks([])
        # Further plot specs
        ylim([-.5,1])

        # Return the handlers
        return (self.fig, self.im_input, 
                self.im_output, self.im_target)




    # Updates images at each frame of the animation
    # data    : list of tuples    Each row contains the
    #                             arguments of update for 
    #                             a frame
    # returns : tuple             The handlers of the 
    #                             images                   
    def update(self, data) :

        # Unpack data
        input_,output_,target_ = data

        # Update data of plot1,  2 and 3
        self.im_input.set_array(to_mat(input_))

        # set data of the 2nd plot
        # (change height of each bar)
        for rect, h in zip(self.im_output, output_ ) :
            rect.set_height(h)
        
        # set data of the 3nd plot
        # (change height of each bar)
        for rect, h in zip(self.im_target, target_ ) : 
            rect.set_height(h)

        # Return the handlers
        return (self.im_input, self.im_output,
                self.im_target) 


    def training_animation(self):

        # Our function to render videos inline 
        # (see anim_to_html.py)
        import anim_to_html as AH

        # The first pattern
        # initialize the figure
        self.init_fig()

        # We use a generator (see https://goo.gl/ekU3u2) to Build 
        # the sequence of update arguments for the three plots.
        data = [ ( self.input_store[:,t], self.output_store[:,t], self.label_store[:,t] ) 
                for t in xrange(self.trials) ]

        # Create and render the animation
        anim = animation.FuncAnimation(self.fig, self.update, data, blit=True)
        return AH.display_animation(anim, filename="mnist-perceptron-training.gif")


