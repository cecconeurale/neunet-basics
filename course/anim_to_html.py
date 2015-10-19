from tempfile import NamedTemporaryFile
import matplotlib.pyplot as plt
import urllib
import base64

IMG_TAG = '<img src = "data:image/png;base64,{0}"/>'

def anim_to_html(anim):
    if not hasattr(anim, '_encoded_gif'):
        with NamedTemporaryFile(suffix='.gif') as f:
            anim.save(f.name, fps = 10, writer='imagemagick')
            gif = open(f.name, "rb").read()
        anim._encoded_gif =  gif.encode("base64")
    
    return IMG_TAG.format(anim._encoded_gif)

from IPython.display import HTML
def display_animation(anim):
    plt.close(anim._fig)
    return HTML(anim_to_html(anim))


