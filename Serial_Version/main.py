import numpy as np 
from scipy import ndimage
import pylab
from SynthTexture import *

if __name__ == "__main__":

	tex=ndimage.imread('rings.jpg')
	tex=tex/255.0;
	#print tex.shape[0], tex.shape[1]
	Sythim=SynthTexture(tex, 13, [50,50])

	pylab.imshow(Sythim)
	pylab.show()
