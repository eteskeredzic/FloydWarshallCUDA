#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define inf 9999 // no edge between vertices (since we don't want to import infinity)

__global__ void gpufun(int n, int k, float* x, int* qx) 
{

    int ix = blockIdx.x * blockDim.x + threadIdx.x;
    int j = ix & (n - 1);
    float tmp = x[ix - j + k] + x[k * n + j];

    if (x[ix] > tmp)  // find lesser of x[ix] and tmp
    {
        x[ix] = tmp;
        qx[ix] = k;
    }
}

int main(int argc, char **argv) 
{

    cudaEvent_t start, stop;

    float *host_A, *host_D, *dev_x, *A, *D, tolerance = 0.001, dt_ms = 0, sum = 0;

    int *host_Q, *dev_qx, *Q, i = 0, j = 0, bk = 0, k = 0, n = atoi(argv[1]), gputhreads = 512;
    char runcpu = argv[2][0];
    double t1s, t2s, t3s, t4s, t5s;

    printf("==========================================\n");
    printf("RUNNING WITH %d NODES \n", n);
    printf("\n");

    cudaMalloc(&dev_x, n * n * sizeof(float));
    cudaMalloc(&dev_qx, n * n * sizeof(float));

    // Arrays for the CPU
    A = (float *) malloc(n * n * sizeof(float)); // original matrix A
    D = (float *) malloc(n * n * sizeof(float)); // original matrix B
    Q = (int *) malloc(n * n * sizeof(int)); // original matrix Q

    // Arrays for the GPU
    host_A = (float *) malloc(n * n * sizeof(float));
    host_D = (float *) malloc(n * n * sizeof(float));
    host_Q = (int *) malloc(n * n * sizeof(int));

    // generate random graph

    srand(time(NULL));
    for (i = 0; i < n; ++i) 
    {
        for (j = 0; j < n; ++j) 
        {
            if (i == j) 
            {
                A[i * n + j] = 0;
            } 
            else 
            {
                A[i * n + j] = 1200 * (float) rand() / RAND_MAX + 1;
                if (A[i * n + j] > 1000) 
                {
                    A[i * n + j] = inf;
                }
            }
            Q[i * n + j] = -1;
            D[i * n + j] = A[i * n + j];
            host_A[i * n + j] = A[i * n + j]; // copy A to host_A
            host_Q[i * n + j] = Q[i * n + j]; // copy Q to host_Q
        }
    }

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // First copy, CPU -> GPU

    cudaEventRecord(start, 0);
    cudaMemcpy(dev_x, host_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_qx, host_Q, n * n * sizeof(int), cudaMemcpyHostToDevice);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt_ms, start, stop);
    printf("Transfer CPU -> GPU, time: %lf ms\n", dt_ms);
    sum+=dt_ms;
    t1s = dt_ms;

    // Calculate parameters for GPU

    bk = (int)(n * n / 512);
    if (bk <= 0) 
    {
        bk = 1;
        gputhreads = n*n;
    } 

    printf("\n");
    printf("Number of CUDA blocks: %d\nNumber of CUDA threads per block: %d \n", bk, gputhreads);
    printf("\n");

    cudaEventRecord(start, 0);
    // start algo on GPU
    for (k = 0; k < n; ++k) 
    {
        gpufun <<<bk, gputhreads>>>(n, k, dev_x, dev_qx);
    }
    cudaDeviceSynchronize(); // wait until all threads are done
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt_ms, start, stop);
    printf("Computation time on GPU: %lf ms\n", dt_ms);

    sum+=dt_ms;
    t2s = dt_ms;

    // Second copy, GPU -> CPU
    cudaEventRecord(start, 0);
    cudaMemcpy(host_D, dev_x, n * n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(host_Q, dev_qx, n * n * sizeof(int), cudaMemcpyDeviceToHost);
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(start); 
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&dt_ms, start, stop);
    printf("Transfer GPU -> CPU, time: %lf ms\n", dt_ms);
    sum+=dt_ms;
    t3s = dt_ms;

    printf("Total time: %lf ms\n\n----------------------------\n", sum);
    t4s = sum;
  
    // Running sequentially on CPU now

    if(runcpu == 'y')
    {
        printf("\n");
        printf("Sequential execution on CPU (could take a while)... \n");
        cudaEventRecord(start, 0);
        for (k = 0; k < n; ++k) 
        {
            for (i = 0; i < n; ++i) 
            {
                for (j = 0; j < n; ++j) 
                {

                    if ((D[i * n + k] + D[k * n + j]) < D[i * n + j]) 
                    {
                        D[i * n + j] = D[i * n + k] + D[k * n + j];
                        Q[i * n + j] = k;
                    }
                }
            }
        }
 
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(start); 
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&dt_ms, start, stop);
        printf("CPU Time: %lf ms\n", dt_ms);
        t5s = dt_ms;
        printf("\n");

        // Result validation

        printf("Comparing CPU results with GPU results...");
        for (i = 0; i < n; ++i) 
        {
            for (j = 0; j < n; ++j) 
            {
                if (abs(D[i * n + j] - host_D[i * n + j]) > tolerance) 
                {
                    printf("ERROR: Different results in row i = %d and column j = %d, CPU result = %f GPU result = %f \n", i, j, D[i * n + j], host_D[i * n + j]);
                    break;
                }
            }
        }
        printf("Comparison complete! \n");
      }
    else
    {
        t5s = -1.0;
    }
    printf("Results are written to file resultsfwreg.csv\n==========================================\n");
    FILE *fptr;
    fptr = fopen("resultsfwreg.csv","a");

    fprintf(fptr,"%d,%d,%d,%lf,%lf,%lf,%lf,%lf\n",n,bk,gputhreads,t1s,t2s,t3s,t4s,t5s);
    fclose(fptr);
    return 0;
}

