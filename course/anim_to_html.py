from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import urllib
import base64

IMG_TAG = '<img src = "data:image/png;base64,{0}"/>'

def anim_to_html(anim, filename = None):
    if not hasattr(anim, '_encoded_gif'):
        if filename == None :
            filename =  NamedTemporaryFile(suffix='.gif').name
        anim.save(filename, fps = 10, writer='imagemagick')
        gif = open(filename, "rb").read()
        anim._encoded_gif =  gif.encode("base64")
    
    return IMG_TAG.format(anim._encoded_gif)
    
    return IMG_TAG.format(anim._encoded_gif)

from IPython.display import HTML
def display_animation(anim, filename = None):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))

def display_animation_in_markup(anim, filename):
    plt.close(anim._fig)
    anim_to_html(anim, filename)
