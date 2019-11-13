#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define block_size 32

#define n 1024


__global__ void mul_matrix(int *a, int *b, int *c){
    int row = threadIdx.y;
    int col = threadIdx.x;
	int my_x = blockIdx.x*blockDim.x + threadIdx.x;
    int my_y = blockIdx.y*blockDim.y + threadIdx.y;
    __shared__ int A_s[32][32];
    __shared__ int B_s[32][32];
    int local_c = 0;
    int i,j;

    for(i = 0; i < n/block_size ;i++)
    {
        A_s[row][col] = a[my_x*n + (i*blockDim.y + col)];
        B_s[row][col] = b[(i*blockDim.x+row)*n + my_y];
        __syncthreads();
        for(j = 0; j < block_size; j++)
        {
            local_c += A_s[row][j] * B_s[j][col];
        }
        __syncthreads();
    }
    c[my_x*n+my_y] = local_c;
}

int main(){		
    int i;
    int *a = (int*)malloc(sizeof(int)*n*n);
    int *b = (int*)malloc(sizeof(int)*n*n);
    int *c = (int*)malloc(sizeof(int)*n*n);
	
	  for(i=0; i<n*n; i++){
			a[i]=1;
			b[i]=2;
            c[i]=0;
  		}
		int *gpu_a, *gpu_b, *gpu_c;
		cudaMalloc((void**)&gpu_a, sizeof(int)*n*n);
		cudaMalloc((void**)&gpu_b, sizeof(int)*n*n);
		cudaMalloc((void**)&gpu_c, sizeof(int)*n*n);
		
		struct timespec start, stop; 
	    double time;
	  
		cudaMemcpy(gpu_a, a, sizeof(int)*n*n, cudaMemcpyHostToDevice);
		cudaMemcpy(gpu_b, b, sizeof(int)*n*n, cudaMemcpyHostToDevice);
		dim3 dimGrid(32,32);
		dim3 dimBlock(32,32);
		
		if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}

		mul_matrix<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);
		
		cudaMemcpy(c, gpu_c, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
		
		if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		printf("time is %f ns\n", time*1e9);	 
		
        printf("C[451][451]=%d\n",c[1024*451 + 451]);

		free(a);
		free(b);
		free(c);
		cudaFree(gpu_a);  
		cudaFree(gpu_b);  
		cudaFree(gpu_c);  
		return 0;
}	