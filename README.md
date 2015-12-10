# Parallel Texture Synthesis
Harvard CS 205 Final Project

Original algorithm by Efros and Leung.  
This simple algorithm works very well to generate plausible texture pattern of various sizes using a small texture patch.  
The authors gave a very detailed discription on their website: http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html  
However, the algorithm is pretty slow because basically it's generating one pixel at a time. Here we're thinking about accelerating it by using OpenCL and let GPU handle the texture synthesis.  

### Parallel Stages
We implement our parallel algorithm in several progressive stages.  


**For more information, you can check our website at http://parallelimageprocessing.weebly.com/!**

We provide two versions of code for download. One is the fully parallelized version with maximum performance improvement.  Another is the original serial version code. 

### Versions
The fully parallelized version consists of two files:  
1. a python driver file named as "parallelized_driver.py"  
2. an OpenCL file named as "filling pixels.cl".  

The original serial version code consists of three files:  
1. a python driver file names as "serial_driver.py"  
2. the first python helper file named as "findmatches.py"  
3. the second python helper file named as "synthtexture.py"   

### How to Run the Code
To run our code, you will need to, at least, have Python 2.7 installed on your local computer, and also the PIL Image package. To run the parallelized version of code, you will need to have OpenCL installed on you computer. Our code could be run on all major operating system including Windows, Mac and Linux. Here, we take Mac as an example and explain the process to run our code.  
1. Run the serial version of code:
    1. Download "serial_driver.py", "findmatches.py" and "synthtexture.py" to a local directory 
    2. Open terminal and cd to the directory containing the files you just downloaded
    3. In the terminal, call python to run the "serial_driver.py" with arguments usage outlined below:
       python serial_driver.py <path_of_input_texture> <name_of_generated_image> <size_of_generated_image>
    4. Finally, the generated image will be saved in current directory with specified name  
    
       For example, if your input texture file is located at "../textures/" and named as "input_texture.jpg". You want to generate a 
       200 * 200 image and name it as "output".
       Then, in your terminal window, you should type:
       python serial_driver.py ../textures/input_texture.jpg output 200
2. Run the parallelized version of code:
    1. Download "parallelized_driver.py" and "filling pixels.cl" to a local directory 
    2. Open terminal and cd to the directory containing the files you just downloaded
    3. In the terminal, call python to run the "parallelized_driver.py" with arguments usage outlined below:
       python parallelized_driver.py <path_of_input_texture> <name_of_generated_image> <size_of_generated_image>
    4. Finally, the generated image will be saved in current directory with specified name

       For example, if your input texture file is located at "../textures/" and named as "input_texture.jpg". You want to generate a 
       200 * 200 image and name it as "output".
       Then, in your terminal window, you should type:
       python parallelized_driver.py ../textures/input_texture.jpg output 200

**Note that the size of the image you can generate may depend on the GPU. Generally we don't recommend generating image over 500 by 500. If you find any bug in our code, please leave a message, thank you!**
