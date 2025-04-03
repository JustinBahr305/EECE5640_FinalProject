TARGET ?= fir

all: $(TARGET)

fir:
	g++ -o fir fir.cpp

fir_omp:
	g++ -fopenmp -o fir_omp fir_omp.cpp

fir_avx:
	g++ -mavx512f -o fir_avx fir_avx.cpp

fir_combo:
	g++ -fopenmp -mavx512f -o fir_combo fir_combo.cpp

clean:
	rm -f fir  fir_omp fir_avx fir_combo