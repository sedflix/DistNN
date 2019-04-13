all:
	nvcc main.cu -g -G -lcublas -lcurand utils/matrix.cu layers/layers.cu -o main
clean:
	rm -rf main
