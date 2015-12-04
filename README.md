# Parallel Texture Synthesis
Harvard CS 205 Final Project

Original algorithm by Efros and Leung.  
This simple algorithm works very well to generate plausible texture pattern of various sizes using a small texture patch.  
The authors gave a very detailed discription on their website: http://graphics.cs.cmu.edu/people/efros/research/EfrosLeung.html  
However, the algorithm is pretty slow because basically it's generating one pixel at a time. Here we're thinking about accelerating it 
through various ways. Mainly by using OpenCL and let GPU handle the texture synthesis.  

