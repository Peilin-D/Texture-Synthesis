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
              __global __read_only float * gpu_texture,
              __global __read_only float * gpu_imfilled,
              __global __read_only float * gpu_Gaussian,
              __local float *workgroup,
              __global __write_only float * outImage,
              __global int * I, //Locations of pixels to fill
              __global int * J,
              int width, int height,
              int buf_w, int buf_h,
              int window_size)
{
    int k=get_global_id(2);
    // get center
    int c_row=I[k];
    int c_col=J[k];
    // get upper left corner
    int cor_row=c_row-(window_size-1)/2;
    int cor_col=c_col-(window_size-1)/2;
    // filling local work group -- template
    for(int row=0;i<window_size;i++)
        for(int col=0;j<window_size;j++)
        {
            workgroup[row*window_size+col]=gpu_Image(get_index(width,height,cor_col+col,cor_row+row));
        }
        


}
