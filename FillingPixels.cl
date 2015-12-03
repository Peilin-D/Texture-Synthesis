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
              __local float * workgroup,  // size of work group should be the same as the size of template window
              __local float * mask,
              __local float * sqDiff,  // square difference
              //__global __write_only float * outImage,
              __global int * I,       //Locations of pixels to fill (row index)
              __global int * J,       //Locations of pixels to fill (col index)
              int width, int height,  // width and height of image
              int tex_w, int tex_h,   // width and height of source texture
              int window_size,        // template window size
              float MaxErr)
{
    //int k=get_global_id(2);
    int k=get_group_id(2);

    int lx=get_local_id(0);
    int ly=get_local_id(1);
    // get template window center location in the image
    int cy=I[k];
    int cx=J[k];

    // get upper left corner of the template 
    int cor_x=cx-(window_size-1)/2;
    int cor_y=cy-(window_size-1)/2;

    // filling template and weighted mask
    // for(int y=0;y<window_size;y++)
    // {
    //     for(int x=0;x<window_size;x++)
    //     {
    //         workgroup[y*window_size+x]=gpu_Image[get_index(width,height,cor_x+x,cor_y+y)];
    //         mask[y*window_size+x]=gpu_imfilled[get_index(width,height,cor_x+x,cor_y+y)]*gpu_Gaussian[y*window_size+x];
    //     }
    // }
    const int idx_1D = ly * get_local_size(0) + lx;  //now local_size actually equals to window size

    if (idx_1D < window_size) {
    for (int row = 0; row < window_size; row++) {
            workgroup[row * window_size+idx_1D] = gpu_Image[get_index(width, height, cor_x + idx_1D, cor_y + row)];
            mask[row * window_size+idx_1D] = gpu_imfilled[get_index(width, height, cor_x + idx_1D, cor_y + row)]*gpu_Gaussian[row * window_size+idx_1D];
        }
    }
    // initialize sqDiff to be all 0
    if(idx_1D<window_size+1)
    {
        for (int row = 0; row < window_size+1; row++) {
            sqDiff[row * window_size+idx_1D] = 0;
        }
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Find Match
    float bestErr=100;
    float bestValue=0;
    for(int oy=0;oy<tex_h-window_size+1;oy++)  // Swipe the template window
    {
        for(int ox=0;ox<tex_w-window_size+1;ox++)
        {
            sqDiff[ly*window_size+lx]=mask[ly*window_size+lx]*(workgroup[ly*window_size+lx]-gpu_texture[(ly+oy)*tex_w+lx+ox])*(workgroup[ly*window_size+lx]-gpu_texture[(ly+oy)*tex_w+lx+ox]);
            barrier(CLK_LOCAL_MEM_FENCE);
            
            //-----------------------parallel reduction--------------------------------//
            for(int j=(window_size+1)/2;j>0;j>>=1)
            {
                if(ly<j)
                    sqDiff[ly*window_size+lx]+=sqDiff[(ly+j)*window_size+lx];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            for(int j=(window_size+1)/2;j>0;j>>=1)
            {
                if(lx<j)
                    sqDiff[lx]+=sqDiff[lx+j];
                barrier(CLK_LOCAL_MEM_FENCE);
            }

            if(sqDiff[0]<bestErr)
            {
                bestErr=sqDiff[0];
                bestValue=gpu_texture[(oy+(window_size-1)/2)*tex_w+ox+(window_size-1)/2];
            }   

            //------------------Multi Threads without parallel reduction-------------// 
            // if(lx==0 && ly==0) // let the first thread do the sum
            // {
            //     float err=0;
            //     for(int y=0;y<window_size;y++)
            //     {
            //         for(int x=0;x<window_size;x++)
            //         {
            //             err+=sqDiff[y*window_size+x];
            //         }
            //     }
            //     if(err<bestErr)
            //     {
            //         bestErr=err;
            //         bestValue=gpu_texture[(oy+(window_size-1)/2)*tex_w+ox+(window_size-1)/2];
            //     }
            // }
            //-----------------------Single Thread----------------------------//    
            // }
            // float err=0;
            // for(int y=0;y<window_size;y++) // Loop over the window
            // {
            //     for(int x=0;x<window_size;x++)
            //     {
            //         err+=mask[y*window_size+x]*(workgroup[y*window_size+x]-gpu_texture[(y+oy)*tex_w+x+ox])*(workgroup[y*window_size+x]-gpu_texture[(y+oy)*tex_w+x+ox]);
            //     }
            // }
            // if(err<bestErr)
            // {
            //     bestErr=err;
            //     bestValue=gpu_texture[(oy+(window_size-1)/2)*tex_w+ox+(window_size-1)/2];
            // }
        }
    }
    if(bestErr<MaxErr)
    {
        gpu_Image[get_index(width,height,cx,cy)]=bestValue;
        gpu_imfilled[get_index(width,height,cx,cy)]=1;
    }
}
