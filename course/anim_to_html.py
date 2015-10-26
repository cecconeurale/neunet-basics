from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import urllib
import base64

IMG_TAG = '<img src = "{}">'
IMG_TAG_INLINE = '<img src = "data:image/png;base64,{}">'

def anim_to_html(anim, filename = None, INLINE = False):
    if filename == None :
        filename =  NamedTemporaryFile(suffix='.gif').name
    anim.save(filename, fps = 10, writer='imagemagick')
    
    if INLINE == True:
        if not hasattr(anim, '_encoded_gif'):
            gif = open(filename, "rb").read()
            anim._encoded_gif =  gif.encode("base64")
        IMG = IMG_TAG_INLINE.format(anim._encoded_gif)
    else :
        IMG =IMG_TAG.format(filename)
    print filename
    return IMG 

from IPython.display import HTML
def display_animation(anim, filename = None):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim, filename))

