This folder contains the following files:
1. README.md
2. makefile
3. fir.cpp
4. fir_omp.cpp
5. fir_avx.cpp
6. fir_combo.cpp
7. fir.script
8. fir_cascadelake.script
9. fir_avx.script
10. PM_lowpass.fda

This folder contains the following directories
1. Filters
2. Input_Samples
3. Output_Samples

Run the command "make TARGET=fir" to generate the following executable:
1. fir

Run the command "make TARGET=fir_omp" to generate the following executable:
1. fir_omp

Run the command "make TARGET=fir_avx" to generate the following executable:
1. fir_avx

Run the command "make TARGET=fir_combo" to generate the following executable:
1. fir_combo

Run the command "make clean" to delete all executables.

To execute the program on a CPU node without AVX support, run the command "sbatch fir.script".
To execute the program on a CPU node with AVX support, run either command below:
"sbatch fir_avx.script" - Runs on any AVX node available
"sbatch fir_cascadelake.script" - Runs on an AVX node in the courses partition
