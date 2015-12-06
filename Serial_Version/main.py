import numpy as np 
from scipy import ndimage
import pylab
from PIL import Image
from SynthTexture import *


if __name__ == "__main__":

	tex=ndimage.imread('../textures/D1.gif')
	tex=tex/255.0;
	#print tex.shape[0], tex.shape[1]
	Sythim=SynthTexture(tex, 13, [50,50])

	im_out=Image.fromarray(Sythim*255)
	im_out.show()
