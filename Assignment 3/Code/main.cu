//-lm to link math.h - optional
#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <ctype.h>
#include <math.h>
__global__ void database_update(int* mat,int* col_comp,int* col_vals,int* upd_vals,int m,int n,int q)
{
    int gid = blockDim.x * blockIdx.x + threadIdx.x;
    if(gid >= m*q*20)
      return;
    int pn = gid/(m*q);
    int z = gid % (m*q);
    int qn = z%q;
    int rn = z/q;
    int col_c = col_comp[2*qn];
    int v = col_comp[2*qn+1];
    int col_upd = col_vals[qn*20+pn];
    int upd_val = upd_vals[qn*20+pn];
    if((mat[n*rn + (col_c-1)] == v) && (col_upd > 0))
      atomicAdd(&mat[n*rn + col_upd-1],upd_val);
}
int main(int argc,char* argv[])
{
    FILE* fp1 = fopen(argv[1],"r");
    FILE* fp2 = fopen(argv[2],"w");
    int m,n,q;
    fscanf(fp1,"%d%d",&m,&n);
    int* arr = (int*)malloc(sizeof(int)*m*n);
    int i,j;
    for(i = 0;i < m;i++)
    {
        for(j = 0;j < n;j++)
        {
            fscanf(fp1,"%d",&arr[i*n + j]);
        }
    }
    fscanf(fp1,"%d",&q);
    int* upd_vals = (int*)malloc(sizeof(int)*q*20);//worst case size of the update array
    int* col_vals = (int*)malloc(sizeof(int)*q*20);
    int* comp_col = (int*)malloc(sizeof(int)*q*2);
    for(i = 0;i < q;i++)
    {
        char c;
        while(fscanf(fp1,"%c",&c))
        {
          if(c == 'U')
            break;
        }
        while(fscanf(fp1,"%c",&c))
        {
          if(c == 'C')
            break;
        }
        int col_num,val,p;
        char op_type;
        fscanf(fp1,"%d%d%d",&col_num,&val,&p);
        comp_col[2*i] = col_num;
        comp_col[2*i+1] = val;
        for(j = 0;j < p;j++)
        {
            int col_to_upd,upd_val;
            while(fscanf(fp1,"%c",&c))
            {
              if(c == 'C')
                break;
            }
            fscanf(fp1,"%d%d",&col_to_upd,&upd_val);
            while(fscanf(fp1,"%c",&op_type))
            {
                if((op_type == '+') || (op_type == '-'))
                  break;
            }
            if(op_type ==  '-')
              upd_val *= -1;
            upd_vals[20*i+j] = upd_val;
            col_vals[20*i+j] = col_to_upd;
        }
        for(j = p;j < 20;j++)
        {
            col_vals[20*i+j] = 0;//no update operations
        }
    }
    int *k1;
    int *k2;
    int* arr1;
    int* col_v;
    cudaMalloc(&k1,sizeof(int)*q*20);
    cudaMalloc(&k2,sizeof(int)*q*20);
    cudaMalloc(&arr1,sizeof(int)*m*n);
    cudaMalloc(&col_v,sizeof(int)*q*2);
    cudaMemcpy(col_v,comp_col,sizeof(int)*q*2,cudaMemcpyHostToDevice);
    cudaMemcpy(arr1,arr,sizeof(int)*m*n,cudaMemcpyHostToDevice);
    cudaMemcpy(k1,upd_vals,sizeof(int)*q*20,cudaMemcpyHostToDevice);
    cudaMemcpy(k2,col_vals,sizeof(int)*q*20,cudaMemcpyHostToDevice);
    int x = ceil((m*q*20.0)/1024);
    database_update<<<x,1024>>>(arr1,col_v,k2,k1,m,n,q);
    cudaDeviceSynchronize();
    cudaMemcpy(arr,arr1,sizeof(int)*m*n,cudaMemcpyDeviceToHost);
    for(i = 0;i < m;i++)
    {
        for(j = 0;j < n;j++)
        {
            fprintf(fp2,"%d ",arr[n*i+j]);
        }
        fprintf(fp2,"\n");
    }
    fclose(fp1);
    fclose(fp2);
    return 0;
}
