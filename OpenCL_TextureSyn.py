# This is CS205 Final Project
import numpy as np
import math
import scipy
#from FindMatches import *
from scipy import ndimage
import pyopencl as cl
import os.path


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
    program = cl.Program(context, open('FindMatches.cl').read()).build(options=['-I', curdir])

    host_texture = ndimage.imread('rings.jpg').astype(np.float32)[::2, ::2].copy()
    host_Synth_Image=np.zeros([100,100])

    gpu_texture = cl.Buffer(context, cl.mem_flags.READ_WRITE, host_texture.size * 4)
   
    # template window size
    w = 13
    halo = (w-1)/2
    global_size = (host_texture.shape[0],host_texture.shape[1])
    local_size = (global_size[0]-2*halo,global_size[1]-2*halo)
    width = np.int32(host_texture.shape[1])
    height = np.int32(host_texture.shape[0])
    # Since the texture is very small, we read it all in to GPU
    local_memory = cl.LocalMemory(4*global_size[0]*global_size[1])
    buf_width = np.int32(global_size[1])
    buf_height = np.int32(global_size[0])

    gpu_validmask = cl.Buffer(context, cl.mem_flags.READ_WRITE, w*w*4)
    gpu_Gaussian = cl.Buffer(context, cl.mem_flags.READ_WRITE, w*w*4)

    cl.enqueue_copy(queue, gpu_texture, host_texture, is_blocking=False)

    MaxErrThreshold=0.3
    sigma=w/6.4
    x,y=np.meshgrid(range(-(w-1)/2,(w-1)/2+1),range(-(w-1)/2,(w-1)/2+1))
    G=(1/(sigma**2*2*np.pi))*np.exp(-(x**2+y**2)/(2*sigma**2))
    nfilled=0
    tofill=synthdim[0]*synthdim[1]-9 # the number of pixels to fill

    cl.enqueue_copy(queue, gpu_Gaussian, G, is_blocking=False)

    synthim=np.ones((synthdim[0],synthdim[1]))*(-1)
    im_filled=np.zeros((synthdim[0],synthdim[1])) # image for testing the original image is filled or not
    offset=(w-1)/2  # offset for zero padding image
    # place the seed in the center and zero padding
    synthim[(math.floor(synthdim[0]/2)-1):(math.floor(synthdim[0])/2+1), (math.floor(synthdim[1]/2)-1):(math.floor(synthdim[1]/2)+1)]=sourceimage[5:7,4:6]
    synthim_padded=np.lib.pad(synthim,(((w-1)/2, (w-1)/2),((w-1)/2, (w-1)/2)),'constant', constant_values=0)
    # find unfilled neighbors
    se= ndimage.generate_binary_structure(2,2)  # use a 3 by 3 structuring element for dilation
    im_filled=(synthim>=0)
    im_dil=ndimage.binary_dilation(im_filled,structure=se)
    [I,J]=np.nonzero(im_dil-im_filled)

    while nfilled<tofill:
        progress=0;
            # get template
        for i in range(0,len(I)):
                template=synthim_padded[I[i]+offset-(w-1)/2:I[i]+offset+(w-1)/2+1,J[i]+offset-(w-1)/2:J[i]+offset+(w-1)/2+1]
                validmask=(template>=0)
                cl.enqueue_copy(queue, gpu_validmask, validmask, is_blocking=False)
                #match_list = FindMatches(template, validmask, sourceimage, G)
                program.FindMatches(queue, global_size, local_size, 
                                    gpu_texture, gpu_validmask, gpu_Gaussian, local_memory, 
                                    match_list,
                                    width, height,
                                    buf_width, buf_height, halo)
                print match_list
                BstInd=np.random.randint(0,len(match_list))
                BestMatch=match_list[BstInd][1]
                MatchErr=match_list[BstInd][0]
                if MatchErr<MaxErrThreshold:
                    synthim_padded[I[i]+offset,J[i]+offset]=BestMatch
                    progress=1
                    nfilled=nfilled+1
                    im_filled[I[i],J[i]]=1
        if progress==0:
                MaxErrThreshold=MaxErrThreshold*1.1
        im_dil=ndimage.binary_dilation(im_filled,structure=se)
        [I,J]=np.nonzero(im_dil-im_filled)
        
    synthim=synthim_padded[offset:offset+synthdim[0],offset:offset+synthdim[1]]


