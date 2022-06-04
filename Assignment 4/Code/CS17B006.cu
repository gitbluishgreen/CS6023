#include <cuda.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#define MAX 10000

struct minhelper
{
  int _a;
  minhelper(int a): _a(a){}
  __host__ __device__ int operator()(const int x,const int y)const{
    return min(MAX*max(_a,x/MAX)+x%MAX,MAX*max(_a,y/MAX)+y%MAX);
  }
};

int schedule(int N,int M,int* arrival_times,int* burst_times,int** cores_schedules,int* cs_lengths)
{
  thrust::device_vector<int> mini_time(M);
  thrust::sequence(thrust::device,mini_time.begin(),mini_time.end(),0);
  int i;
  for(i = 0;i < M;i++)
  {
      cores_schedules[i] = (int*)malloc(sizeof(int)*N);
      cs_lengths[i] = 0;
  }
  int turnaround_time = 0;
  int* x = (int*)malloc(sizeof(int));
  for(i = 0;i < N;i++)
  {
    int at = arrival_times[i];
    int b = burst_times[i];
    int ind1 = thrust::reduce(mini_time.begin(),mini_time.end(),(int)1000000000,minhelper(at));
    int ind = ind1 % MAX;
    turnaround_time += max((ind1 / MAX) - at,0) + b;
    cores_schedules[ind][cs_lengths[ind]] = i;
    cs_lengths[ind]++;
    int* x1 = thrust::raw_pointer_cast(&mini_time[ind]);
    *x = ind + (ind1/MAX + b) * MAX;
    cudaMemcpy(x1,x,sizeof(int),cudaMemcpyHostToDevice);
  }
  return turnaround_time;
}
