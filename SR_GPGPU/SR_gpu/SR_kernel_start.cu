//extern int *test;

void SR_kernel_start(int w, int h, int ww, int hh){
	//cudaMalloc((void**)&test, 100*sizeof(int));
	/*
	cudaMalloc((void**)&d_ansR, ww*hh*sizeof(int));
	cudaMalloc((void**)&d_ansG, ww*hh*sizeof(int));
	cudaMalloc((void**)&d_ansB, ww*hh*sizeof(int));

	cudaFree(d_ansR);
	cudaFree(d_ansG);
	cudaFree(d_ansB);
	*/
}

void SR_kernel_end(){
	//cudaFree(test);
}

