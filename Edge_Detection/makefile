all: edge edge_tiled edge_uchar3

edge:
	nvcc -o edge edge.cpp sobel.cu

edge_tiled:
	nvcc -o edge_tiled edge.cpp tiled_sobel.cu

edge_uchar3:
	nvcc -o edge_uchar3 edge.cpp uchar3_sobel.cu

clean:
	rm edge edge_tiled edge_uchar3

