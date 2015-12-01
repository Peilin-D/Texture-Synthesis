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
FindMatches(__global __read_only float * gpu_texture,
            __global __read_only float * gpu_validmask,
            __global __read_only float * gpu_Gaussian,
            __global __read_only float * template,
            //__global __read_only float * local_memory,
            __local float *buffer,
            __global __write_only float * match_list,
            int width, int width,
            int buf_w, int buf_h,
            const int halo)
{
    // Note: It may be easier for you to implement median filtering
    // without using the local buffer, first, then adjust your code to
    // use such a buffer after you have that working.
    
    //get global location
    const int x = get_global_id(0);
    const int y = get_global_id(1);

    //get local locatoin
    const int lx = get_local_id(0);
    const int ly = get_local_id(1);

    //1-D index of local location
    const int index = ly * get_local_size(0) + lx;

    //get corner location of buffer

    const int buf_corner_x = x - lx - halo;
    const int buf_corner_y = y - ly - halo;

    //get location of buffer
    const int buf_x = lx + halo;
    const int buf_y = ly + halo;

    //code refered to load_halo.cl
    if (index < buf_w) {// From load_halo.cl
        for (int row = 0; row < buf_h; row++) {
            buffer[row * buf_w + index] = gpu_texture[get_index(w, h, buf_corner_x + index, buf_corner_y + row)];
        }
    }

    // Compute 3x3 median for each pixel in core (non-halo) pixels
    //
    // We've given you median9.h, and included it above, so you can
    // use the median9() function.
    
    barrier(CLK_LOCAL_MEM_FENCE);
    
    const int window_width = 2*halo+1;
    int offset_x = buf_x-(window_width-1)/2;
    int offset_y = buf_y-(window_width-1)/2;
    if ((x < w) && (y < h)) {
        for(int row = buf_x-(window_width-1)/2; row<buf_x-(window_width-1)/2 + window_width; row++){
            for(int col=buf_y-(window_width-1)/2; col<buf_y-(window_width-1)/2 + window_width;col++){
                match_list[y*w+x]=buffer[get_index(buf_w, buf_h, col, row)]-template[col-offset_x, row-offset_y];
    }   
    // Each thread in the valid region (x < w, y < h) should write
    // back its 3x3 neighborhood median.
}