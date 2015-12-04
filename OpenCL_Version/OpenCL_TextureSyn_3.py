# This is CS205 Final Project
import numpy as np
from PIL import Image
import scipy
from scipy import ndimage
import pyopencl as cl
import os.path
import pylab


if __name__ == '__main__':
    # List our platforms
    platforms = cl.get_platforms()
    print 'The platforms detected are:'
    print '---------------------------'
    for platform in platforms:
        print platform.name, platform.vendor, 'version:', platform.version

    # List devices in each platform
    for platform in platforms:
        print 'The devices detected on platform', platform.name, 'are:'
        print '---------------------------'
        for device in platform.get_devices():
            print device.name, '[Type:', cl.device_type.to_string(device.type), ']'
            print 'Maximum clock Frequency:', device.max_clock_frequency, 'MHz'
            print 'Maximum allocable memory size:', int(device.max_mem_alloc_size / 1e6), 'MB'
            print 'Maximum work group size', device.max_work_group_size
            print '---------------------------'

    # Create a context with all the devices
    devices = platforms[0].get_devices()
    context = cl.Context(devices)
    print 'This context is associated with ', len(context.devices), 'devices'

    # Create a queue for transferring data and launching computations.
    # Turn on profiling to allow us to check event times.
    queue = cl.CommandQueue(context, context.devices[0],
                            properties=cl.command_queue_properties.PROFILING_ENABLE)
    print 'The queue is using the device:', queue.device.name

    curdir = os.path.dirname(os.path.realpath(__file__))
    program = cl.Program(context, open('FillingPixels.cl').read()).build(options='')#(options=['-I', curdir])


    host_texture = ndimage.imread('rings.jpg').astype(np.float32)
    # im_c=np.zeros([host_texture.shape[0],host_texture.shape[1],3])
    # im_c[:,:,1]=host_texture
    # im=Image.fromarray(im_c)
    # im.show()

    host_texture = host_texture/255



    # template window size
    w = 15
    synthdim=[50,50]
    tex_width = np.int32(host_texture.shape[1])
    tex_height = np.int32(host_texture.shape[0])
    
    MaxErrThreshold=0.3
    sigma=w/6.4
    x,y=np.meshgrid(range(-(w-1)/2,(w-1)/2+1),range(-(w-1)/2,(w-1)/2+1))
    G=(1/(sigma**2*2*np.pi))*np.exp(-(x**2+y**2)/(2*sigma**2)).astype(np.float32)
    nfilled=0
    tofill=synthdim[0]*synthdim[1]-9 # the number of pixels to fill

    synthim=(np.ones((synthdim[0],synthdim[1]))*(-1)).astype(np.float32)
    #im_filled=np.zeros((synthdim[0],synthdim[1])).astype(np.float32) # image for testing the original image is filled or not

    # place the seed in the center and zero padding
    synthim[(np.floor(synthdim[0]/2)-1):(np.floor(synthdim[0])/2+2), (np.floor(synthdim[1]/2)-1):(np.floor(synthdim[1]/2)+2)]=host_texture[5:8,4:7]
 
    # GPU Global Buffers
    #gpu_im_not_filled = cl.Buffer(context, cl.mem_flags.READ_WRITE, synthim.size*4)
    gpu_imfilled = cl.Buffer(context, cl.mem_flags.READ_WRITE, synthim.size*4)
    gpu_texture = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_texture.size * 4)
    gpu_Gaussian = cl.Buffer(context, cl.mem_flags.READ_WRITE, w*w*4)
    gpu_Image = cl.Buffer(context, cl.mem_flags.READ_WRITE, synthim.size * 4)


    cl.enqueue_copy(queue, gpu_texture, host_texture) #is_blocking=False)
    cl.enqueue_copy(queue, gpu_Image, synthim) #is_blocking=False)
    cl.enqueue_copy(queue, gpu_Gaussian, G) #is_blocking=False)
    
    # find unfilled neighbors
    #se= ndimage.generate_binary_structure(2,2)  # use a 3 by 3 structuring element for dilation
    im_filled=(synthim>=0).astype(np.float32)
    im_not_filled=(synthim<0).astype(np.float32)
    [I,J]=np.nonzero(im_not_filled)

    # sort I and J by the distance to image center
    ImCenX=np.floor(synthdim[0]/2)
    ImCenY=np.floor(synthdim[1]/2)
    vecImCenX=np.ones(len(I))*ImCenX
    vecImCenY=np.ones(len(I))*ImCenY
    vecDist=np.sqrt((I-vecImCenX)**2+(J-vecImCenY)**2)
    DistTup=np.array(zip(I,J,vecDist),dtype=[('row',int),('col',int),('dist',float)])
    sortedInd=np.argsort(DistTup,order='dist')
    I=I[sortedInd]
    J=J[sortedInd]
    I=np.int32(I)
    J=np.int32(J)
    gpu_I = cl.Buffer(context, cl.mem_flags.READ_WRITE, len(I)*4)
    gpu_J = cl.Buffer(context, cl.mem_flags.READ_ONLY, len(I)*4)


    #cl.enqueue_copy(queue, gpu_im_not_filled, im_not_filled)
    cl.enqueue_copy(queue, gpu_imfilled, im_filled) #is_blocking=False)
    cl.enqueue_copy(queue, gpu_I, I) #is_blocking=False)
    cl.enqueue_copy(queue, gpu_J, J) #is_blocking=False)
    
    # GPU Local work group
    workgroup = cl.LocalMemory(4*w*w)
    mask = cl.LocalMemory(4*w*w)
    sqDiff = cl.LocalMemory(4*w*w)
    texture=cl.LocalMemory(4*tex_width*tex_height)

    # global size and local size
    #global_size = (1,1,len(I))
    local_size = (w,w,1)

    # Call Kernel
    n=1
    flag=True
    while flag:
        if nfilled+8*n+8<=tofill:
            global_size=(w,w,8*n+8)
        else:
            global_size=(w,w,tofill-nfilled)
            flag=False
        program.FillingPixels_v5(queue, global_size, local_size,
                                gpu_Image, gpu_texture, gpu_imfilled, gpu_Gaussian,
                                workgroup, mask, sqDiff, texture, gpu_I, gpu_J, 
                                np.int32(synthdim[0]),np.int32(synthdim[1]),np.int32(tex_width),np.int32(tex_height),
                                np.int32(w),np.float32(MaxErrThreshold),np.int32(n))
        nfilled=nfilled+8*n+8
        n=n+1
    cl.enqueue_copy(queue, synthim, gpu_Image)

    out_im=Image.fromarray(synthim*255)
    out_im.show()
    #pylab.imshow(synthim)
    #pylab.show()


