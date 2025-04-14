This folder contains the following files:
1. README.md
2. makefile
3. edge.cpp
4. sobel.cu
5. tiled_sobel.cu
6. uchar3_sobel.cu
7. edge.script

This folder contains the following directories
1. Input_Samples
2. Output_Samples

Run the command "make" to generate the following executables:
1. edge
2. edge_tiled
3. edge_uchar3

Run the command "make clean" to delete all executables.

To execute the program on a P100 GPU node, run the command "sbatch edge.script".