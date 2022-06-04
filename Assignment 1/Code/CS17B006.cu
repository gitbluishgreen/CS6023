__global__ void per_row_kernel(int* in,int N)
{
    int blockId = blockIdx.x * blockDim.y + threadIdx.y;
    int rn = blockDim.x * blockId + threadIdx.x;
    if(rn >= N)
      return; 
    int i;
    for(i = 0;i < N;i++)
    {
        int old_ind = N*rn + i;
        int new_ind = N*i + rn;
        if(rn < i)
        {
            int t = in[old_ind];
            in[old_ind] = in[new_ind];
            in[new_ind] = t;
        }
    }
}

__global__ void per_element_kernel(int *in,int N)
{
    int blockId = blockIdx.x + gridDim.x * (blockIdx.y + (gridDim.y * blockIdx.z));
    int ind = blockId * blockDim.x + threadIdx.x;
    int x = ind / N;
    int y = ind % N;
    if(x < y)
    {
        int t = in[N*x + y];
        in[N*x + y] = in[N*y + x];
        in[N*y + x] = t;
    }
}

__global__ void per_element_kernel_2D(int* in,int N)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x;
    int ind = blockDim.x *((blockId * blockDim.y) + threadIdx.y) + threadIdx.x;
    if(ind >= N*N)
      return;
    int x = ind / N;
    int y = ind % N;
    if(x < y)
    {
        int t = in[N*x + y];
        in[N*x + y] = in[N*y + x];
        in[N*y + x] = t;
    }

}
