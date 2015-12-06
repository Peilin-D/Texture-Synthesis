import numpy as np 
from scipy import ndimage
import pylab
from PIL import Image
from SynthTexture import *
import time


if __name__ == "__main__":
    
    
    t0=time.time()
	tex=ndimage.imread('../textures/D1.gif')
	tex=tex/255.0;
	#print tex.shape[0], tex.shape[1]
	Sythim=SynthTexture(tex, 13, [200,200])
    print "Serial Version ", time.time()-t0, " seconds"
	im_out=Image.fromarray(Sythim*255)
	im_out.show()
