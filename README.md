# FloydWarshallCUDA
 A CUDA implementation of the Floyd Warshall APSP algorithm (both regular and blocked versions)
 
## Contents:

This repository contains:
- An implementation of a regular (naive) parallel Floyd-Warhsall algorithm (folder: regular);
- An implementation of a blocked parallel Floyd-Warshall algorithm (folder: blocked), based on the work done by Katz and Kider in their paper "All-pairs shortest-paths for large graphs on the GPU", available at https://dl.acm.org/doi/10.5555/1413957.1413966;
- Both implementation contain BASH scripts which can be used to run the code;
- Some testing results (folder: img);

## Compiling:
To compile, enter the following command in your terminal (the same commands are used with the blocked version, just replace 'fwreg' with 'fwblocked'):

`nvcc fwreg.cu -o fwreg -arch compute_50 -code compute_50`

for Maxwell architecture, or

`nvcc fwreg.cu -o fwreg -arch compute_60 -code compute_60`

for Pascal architecture. You can also compile it with other architectures (see https://docs.nvidia.com/cuda/cuda-compiler-driver-nvcc/index.html#virtual-architecture-feature-list)

## Running:
To run, you can simply type `./<filename> NUM_VERTICES RUN_ON_CPU` (example: `./fwreg 1024 y`). The second parameter indicates that you also want to preform the algorithm in it's sequential form on the CPU (that can take a lot of time for large graphs, so you can disable it with this parameter).

If instead you want to run it for a series of tests, you can use the bash script `runTestsfwreg.sh` and `runTestsfwblocked.sh`, where the arguments you specify are the highest number of vertices that you want to run, and if you want to run the sequential code (example: `./runTestsfwreg.sh 1024 n` will run tests for 128, 256, 512, and 1024 vertices, but only on the GPU - the sequential variant on the CPU will be skipped). 

All results will be written to the file `resultsfwreg.csv` (`resultsfwblocked.csv` for the blocked variant). If you have decided to not run the sequential part, then the time -1 sec will be written in the CPU time column.


## Results:
We tested the performance on two GPUs, the NVIDIA GeForce GTX 850M (commerical laptop GPU), and the NVIDIA Jetson TX2 (embedded computer). We also compared the results to a CPU executing sequential code (Intel Core i5-5200U).

The results for execution time are given below:


![CPU vs GTX 850M](https://github.com/eteskeredzic/FloydWarshallCUDA/blob/master/img/graphic1.pdf)

![CPU vs Jetson TX2](https://github.com/eteskeredzic/FloydWarshallCUDA/blob/master/img/graphic1.pdf)

![GTX 850M vs Jetson TX2](https://github.com/eteskeredzic/FloydWarshallCUDA/blob/master/img/graphic1.pdf)

We also tested the energy consumption on the embedded GPU, the results are listed below:


![Peak power consumption during execution](https://github.com/eteskeredzic/FloydWarshallCUDA/blob/master/img/power1.pdf)

![Total energy consumed](https://github.com/eteskeredzic/FloydWarshallCUDA/blob/master/img/energy2.pdf)


```
/*  
 ----------------------------------------------------------------------------  
 "THE BEER-WARE LICENSE" (Revision 42):  
 /eteskeredzic and /kkarahodzi1 wrote this file.  As long as you retain this notice you  
 can do whatever you want with this stuff. If we meet some day, and you think  
 this stuff is worth it, you can buy me a beer in return.   
	/eteskeredzic and /kkarahodzi1 
 ----------------------------------------------------------------------------  
 */  
 ```
