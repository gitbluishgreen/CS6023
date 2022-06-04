#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <math.h>
__global__ void sumRandC(int *a,int* b,int m,int n,int p,int q,int k)
{
 int beg_id = k*(blockDim.x*blockIdx.x + threadIdx.x);
 int i = 0;
 while((i < k) && (beg_id < m*n))
 {
     i++;
    int r = beg_id / n;
    int c = beg_id % n;
    int x = a[beg_id];
    int *ptr1 = b + (q*(r+1)-1);
    int *ptr2 = b + (q*(p-1) + c);
    atomicAdd(ptr1,x);
    atomicAdd(ptr2,x);
    beg_id++;
 }
}

__global__ void findMin(int *a,int *b,int m,int n,int p,int q,int k)
{
 int beg_id = k*(blockDim.x*blockIdx.x + threadIdx.x);
 int i = 0;
 int *ptr = &b[p*q-1];
 while((i < k) && (beg_id < m*n))
 {
    int r = beg_id / n;
    int c = beg_id % n;
    if((r == m-1) || (c == n-1))
    {
        if(r == m-1)
          r++;
        else
          c++;
      int x = b[r*q + c];
      atomicMin(ptr,x);
    }
    i++;
    beg_id++;
 }
}

__global__ void updateMin(int *a,int *b,int m,int n,int p,int q,int k)
{
 int beg_id = k*(blockDim.x*blockIdx.x + threadIdx.x);
 int i = 0;
 int mini = b[p*q-1];
 while((i < k) && (beg_id < m*n))
 {
     i++;
    int r = beg_id / n;
    int c = beg_id % n;
    b[r*q + c] += mini;
    beg_id++;
 }
}
int main(void)
{
    int m,n,k;
    scanf("%d%d%d",&m,&n,&k);
    int* arr = new int[m*n];
    int* arr1 = new int[(m+1)*(n+1)];
    int i,j;
    for(i = 0;i < m;i++)
    {
        for(j = 0;j < n;j++)
        {
          scanf("%d",&arr[n*i+j]);
          arr1[(n+1)*i + j] = arr[n*i+j];
        }
    }
    int t = (n+1)*m;
    for(i = 0;i <= n;i++)
    {
        arr1[t + i] = 0;
    }
    for(i = 0;i <= m;i++)
    {
        arr1[(n+1)*i + n] = 0;
    }
    arr1[(m+1)*(n+1)-1] = INT_MAX;
    int* a;
    cudaMalloc(&a,n*m*sizeof(int));
    cudaMemcpy(a,arr,n*m*sizeof(int),cudaMemcpyHostToDevice);
    int* b;
    cudaMalloc(&b,(n+1)*(m+1)*sizeof(int));
    cudaMemcpy(b,arr1,(n+1)*(m+1)*sizeof(int),cudaMemcpyHostToDevice);
    int tn = (m*n)/k;
    int bn = 1;
    if(tn > 1024)
    {
        bn = ceil(((double)tn)/1024);
        tn = 1024;
    }
    sumRandC<<<bn,tn>>>(a,b,m,n,m+1,n+1,k);
    cudaDeviceSynchronize();
    findMin<<<bn,tn>>>(a,b,m,n,m+1,n+1,k);
    cudaDeviceSynchronize();
    updateMin<<<bn,tn>>>(a,b,m,n,m+1,n+1,k);
    cudaDeviceSynchronize();
    cudaMemcpy(arr1,b,(n+1)*(m+1)*sizeof(int),cudaMemcpyDeviceToHost);
     for(i = 0;i <= m;i++)
     {
         for(j = 0;j <= n;j++)
         {
             printf("%d ",arr1[(n+1)*i+j]);
         }
        printf("\n");
     }
    return 0;
}