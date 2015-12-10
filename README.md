# Parallel Texture Synthesis
Harvard CS 205 Final Project  

Original algorithm by Efros and Leung.  
This simple algorithm works very well to generate plausible texture pattern of various sizes using a small texture patch.  
However, the algorithm is pretty slow because basically it's generating one pixel at a time. Here we're thinking about accelerating it by using OpenCL and let GPU handle the texture synthesis.  

**For detailed description of both serial and parallel algorith, please check our website at http://parallelimageprocessing.weebly.com/!**  

### Parallel Stages
We implement our parallel algorithm in several progressive stages.  
#### Parallel Setup
At very beginning, we adapt our serial implementation using OpenCL, and we send GPU the locations of pixels to be filled, the source texture, the image to be filled and the binary image indicating which pixels have already been filled. But we use only one thread for each work group to calculate the sum square difference (SSD).
#### Stage 1: Multi-threads in Work Group
Next we assign multiple threads to each work group, which is the same size as our template window. So when calculating the SSD over the source texture, in every sliding window, each thread calculates its own difference, and then we use parallel reduction to sum up all the differences in one work group. 
#### Stage 2: Let GPU do all the work
To let GPU do as much work as possible, rather than let CPU calculates which part of pixels are to be filled next by doing image dilation, we send all the unfilled pixels to GPU at once, and then sort the array of locations by the distance to the seed, which is at the center of the synthetic image. And then we generate the image circle by circle in GPU.
#### Stage 3: Copy source texture to Local Buffer
For small source texture, we can load it into local buffer. The source texture would be read several times in the GPU, so we may expect better performance by doing this. But it may not always bring improvement because every thread needs to load it, which would also cost time. A good balance is necessary to find.  

### Versions
We provide two main versions of code. One is the parallel version, another is the original serial version code. 
The parallel code consists of several versions:  

1. an OpenCL file named as "filling pixels.cl", in which there are several versions of implementations
    1. FillingPixels_v1 implements the Parallel Setup Stage, which simply uses single thread in one work group.
    2. FillingPixels_v2 implements Stage 1. 
    3. FillingPixels_v3 implements Stage 2. 
    4. FillingPixels_v4 implements Stage 3. 
2. several python driver files:  
    1. "Parallel_driver.py" uses FillingPixels_v2
    2. "Parallel_driver_2.py" uses FillingPixels_v3
    3. "Parallel_driver_3.py" uses FillingPixels_v4
 
The original serial version code consists of three files:

1. a python driver file names as "serial_driver.py"
2. the first python helper file named as "findmatches.py"  
3. the second python helper file named as "synthtexture.py"   

### How to Run the Code
To run our code, you will need to, at least, have Python 2.7 installed on your local computer, and also the PIL Image package. To run the parallelized version of code, you will need to have OpenCL installed on you computer. Our code should be able to run on all major operating system including Windows, Mac and Linux. Here, we take Mac as an example and explain the process to run our code.  
**Note that when running parallel versions, the system will ask you which platform and device you'd like to use**

1. Run the serial version of code:
    1. Download "serial_driver.py", "findmatches.py" and "synthtexture.py" to a local directory 
    2. Open terminal and cd to the directory containing the files you just downloaded
    3. In the terminal, call python to run the "serial_driver.py" with arguments usage outlined below:  
       `python serial_driver.py <path_of_input_texture> <name_of_generated_image> <template window size> <size_of_generated_image>`
    4. Finally, the generated image will be saved in current directory with specified name  
    
       For example, if your input texture file is located at "../textures/" and named as "input_texture.jpg". You want to generate a 
       200 * 200 image and name it as "output".
       Then, in your terminal window, you should type:  
       `python serial_driver.py ../textures/input_texture.jpg output 15 200 200`

2. Run the parallelized version of code:
    1. Download "Parallel_driver.py" and "filling pixels.cl" to a local directory 
    2. Open terminal and cd to the directory containing the files you just downloaded
    3. In the terminal, call python to run the "parallelized_driver.py" with arguments usage outlined below:  
       `python Parallel_driver.py <path_of_input_texture> <name_of_generated_image> <template window size> <size_of_generated_image>`
    4. Finally, the generated image will be saved in current directory with specified name

       For example, if your input texture file is located at "../textures/" and named as "input_texture.jpg". You want to generate a 
       200 * 200 image and name it as "output".
       Then, in your terminal window, you should type:  
       `python Parallel_driver.py ../textures/input_texture.jpg output 15 200 200`

**Note that the size of the image you can generate may depend on the GPU. Generally we don't recommend generating image over 500 by 500. If you find any bug in our code, please leave a message, thank you!**
