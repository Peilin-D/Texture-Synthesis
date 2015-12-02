int get_index(int w, int h, int x, int y){
    if(x < 0){
        x = 0;
    }
    if(x >= w){
        x = w - 1;
    }
    if(y >= h){
        y = h - 1;
    }
    if(y < 0){
        y = 0;
    }
    return y*w+x;
}

__kernel void
FillingPixels(__global float * gpu_Image,
              __global float * gpu_texture,
              __global float * gpu_imfilled, // binary image for recording which pixels are filled and which are not 
              __global float * gpu_Gaussian, // Guassian weight, same size as template window
              __local float * workgroup,
              __local float * mask,
              //__global __write_only float * outImage,
              __global int * I,       //Locations of pixels to fill (row index)
              __global int * J,       //Locations of pixels to fill (col index)
              int width, int height,  // width and height of image
              int tex_w, int tex_h,   // width and height of source texture
              int window_size,       // template window size
              float MaxErr)
{
    int k=get_global_id(2);

    // get template window center location in the image
    int cy=I[k];
    int cx=J[k];

    // get upper left corner of the template 
    int cor_x=cx-(window_size-1)/2;
    int cor_y=cy-(window_size-1)/2;

    // filling template and weighted mask
    for(int y=0;y<window_size;y++)
    {
        for(int x=0;x<window_size;x++)
        {
            workgroup[y*window_size+x]=gpu_Image[get_index(width,height,cor_x+x,cor_y+y)];
            mask[y*window_size+x]=gpu_imfilled[get_index(width,height,cor_x+x,cor_y+y)]*gpu_Gaussian[y*window_size+x];
        }
    }

    // Find Match
    float bestErr=100;
    float bestValue=0;
    for(int oy=0;oy<tex_h-window_size+1;oy++)  // Swipe the template window
    {
        for(int ox=0;ox<tex_w-window_size+1;ox++)
        {
            float err=0;
            for(int y=0;y<window_size;y++) // Loop over the window
            {
                for(int x=0;x<window_size;x++)
                {
                    err+=mask[y*window_size+x]*(workgroup[y*window_size+x]-gpu_texture[(y+oy)*tex_w+x+ox])*(workgroup[y*window_size+x]-gpu_texture[(y+oy)*tex_w+x+ox]);
                }
            }
            if(err<bestErr)
            {
                bestErr=err;
                bestValue=gpu_texture[(oy+(window_size-1)/2)*tex_w+ox+(window_size-1)/2];
            }
        }
    }
    if(bestErr<MaxErr)
    {
        gpu_Image[get_index(width,height,cx,cy)]=bestValue;
        gpu_imfilled[get_index(width,height,cx,cy)]=1;
    }
}
