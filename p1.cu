#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>
#include <time.h>

#define n 1024


__global__ void matrix_mul(int *a, int *b, int *c){
    int my_x;
    int my_y;
	my_x = blockIdx.x*blockDim.x + threadIdx.x;
    my_y = blockIdx.y*blockDim.y + threadIdx.y;
    int local_c = 0;
    int i;
    for(i = 0; i < n;i++)
    {
        local_c += a[my_x*n+i]*b[i*n+my_y];
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
		
		dim3 dimGrid(64,64);
		dim3 dimBlock(16,16);
		
		if( clock_gettime( CLOCK_REALTIME, &start) == -1 ) { perror( "clock gettime" );}
		
		matrix_mul<<<dimGrid, dimBlock>>>(gpu_a, gpu_b, gpu_c);				
		cudaMemcpy(c, gpu_c, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
		
		if( clock_gettime( CLOCK_REALTIME, &stop) == -1 ) { perror( "clock gettime" );}	  
		time = (stop.tv_sec - start.tv_sec)+ (double)(stop.tv_nsec - start.tv_nsec)/1e9;
		printf("Execution time %f ns\n", time*1e9);	 
		
        printf("C[451][451]= %d\n", c[1024*451 + 451]);
  	
		free(a);
		free(b);
		free(c);
		cudaFree(gpu_a);  
		cudaFree(gpu_b);  
		cudaFree(gpu_c);  
		return 0;
}	