
texture<int, 1, cudaReadModeElementType> TR;
texture<int, 1, cudaReadModeElementType> TG;
texture<int, 1, cudaReadModeElementType> TB;
/*
texture<float, 1, cudaReadModeElementType> Td0;
texture<float, 1, cudaReadModeElementType> Td1;
*/
__constant__ float d_d0[3];
__constant__ float d_d1[3];

//extern __shared__ int row[];

extern "C" void set_filter(float *d0, float *d1){
	cudaMemcpyToSymbol(d_d0, d0, 3 * sizeof(float));
	cudaMemcpyToSymbol(d_d1, d1, 3 * sizeof(float));
}

__device__ void setup_col(int *row0, int *row1, int ww, int h, int index, int *ans){
	row0[0]=(int)(d_d0[1]*ans[index]+d_d0[2]*ans[ww +index]);
	row1[0]=(int)(d_d1[1]*ans[index]+d_d1[2]*ans[ww +index]);
	row0[h-1]=(int)(d_d0[0]*ans[(h-2)*ww +index]+d_d0[1]*ans[(h-1)*ww +index]);
	row1[h-1]=(int)(d_d1[0]*ans[(h-2)*ww +index]+d_d1[1]*ans[(h-1)*ww +index]);

	#pragma unroll
	for(int i=1; i<h-1; ++i){
		row0[i]=(int)(d_d0[0]*ans[(i-1)*ww +index]
					 +d_d0[1]*ans[i*ww +index]
					 +d_d0[2]*ans[(i+1)*ww +index]);
		row1[i]=(int)(d_d1[0]*ans[(i-1)*ww +index]
					 +d_d1[1]*ans[i*ww +index]
					 +d_d1[2]*ans[(i+1)*ww +index]);		
	}
}

__device__ void setup_row(int *row0, int *row1, int w, int index, texture<int, 1, cudaReadModeElementType> rgb){
	/*
	if(tid==0){
		row[0]=(int)(d_d0[1]*tex1Dfetch(rgb, index*w +0)
					+d_d0[2]*tex1Dfetch(rgb, index*w +1));
		row[w]=(int)(d_d1[1]*tex1Dfetch(rgb, index*w +0)
					+d_d1[2]*tex1Dfetch(rgb, index*w +1));
	}
	else if(tid==w-1){
		row[w-1]=(int)(d_d0[0]*tex1Dfetch(rgb, index*w +w-2)
					+d_d0[1]*tex1Dfetch(rgb, index*w +w-1));
		row[w+w-1]=(int)(d_d1[0]*tex1Dfetch(rgb, index*w +w-2)
					+d_d1[1]*tex1Dfetch(rgb, index*w +w-1));
	}
	else{
		row[tid]=(int)(d_d0[0]*tex1Dfetch(rgb, index*w +tid-1)
						 +d_d0[1]*tex1Dfetch(rgb, index*w +tid)
						 +d_d0[2]*tex1Dfetch(rgb, index*w +tid+1));
		row[w+tid]=(int)(d_d1[0]*tex1Dfetch(rgb, index*w +tid-1)
						 +d_d1[1]*tex1Dfetch(rgb, index*w +tid)
						 +d_d1[2]*tex1Dfetch(rgb, index*w +tid+1));
	}*/
	
	row0[0]=(int)(d_d0[1]*tex1Dfetch(rgb, index*w +0)
					+d_d0[2]*tex1Dfetch(rgb, index*w +1));
	row1[0]=(int)(d_d1[1]*tex1Dfetch(rgb, index*w +0)
					+d_d1[2]*tex1Dfetch(rgb, index*w +1));
	row0[w-1]=(int)(d_d0[0]*tex1Dfetch(rgb, index*w +w-2)
					+d_d0[1]*tex1Dfetch(rgb, index*w +w-1));
	row1[w-1]=(int)(d_d1[0]*tex1Dfetch(rgb, index*w +w-2)
					+d_d1[1]*tex1Dfetch(rgb, index*w +w-1));

	#pragma unroll
	for(int i=1; i<w-1; ++i){
		row0[i]=(int)(d_d0[0]*tex1Dfetch(rgb, index*w +i-1)
						 +d_d0[1]*tex1Dfetch(rgb, index*w +i)
						 +d_d0[2]*tex1Dfetch(rgb, index*w +i+1));
		row1[i]=(int)(d_d1[0]*tex1Dfetch(rgb, index*w +i-1)
						 +d_d1[1]*tex1Dfetch(rgb, index*w +i)
						 +d_d1[2]*tex1Dfetch(rgb, index*w +i+1));		
	}
	
}

__global__ void run_col(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<ww){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		int index=round+tid;
		for(int i=0; i<h; ++i){ // compute weight
			R_ori+=ans_R[i*ww +index];
			G_ori+=ans_G[i*ww +index];
			B_ori+=ans_B[i*ww +index];
		}

		int row0[720];
		int row1[720];
		// red
		setup_col(row0, row1, ww, h, index, ans_R);
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_R[i*ww +index]=row0[3*i/2];
			else ans_R[i*ww +index]=row1[3*(i-1)/2 +2];
			e_aft+=ans_R[i*ww +index];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		for(int i=0; i<hh; ++i){
			ans_R[i*ww +index]=(int)((float)ans_R[i*ww +index]/R_rate);
			//if(ans_R[i*ww +index]>255) ans_R[i*ww +index]=255;
			//else if(ans_R[i*ww +index]<0) ans_R[i*ww +index]=0;
		}
		// green
		setup_col(row0, row1, ww, h, index, ans_G);
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_G[i*ww +index]=row0[3*i/2];
			else ans_G[i*ww +index]=row1[3*(i-1)/2 +2];
			e_aft+=ans_G[i*ww +index];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		for(int i=0; i<hh; ++i){
			ans_G[i*ww +index]=(int)((float)ans_G[i*ww +index]/G_rate);
			//if(ans_G[i*ww +index]>255) ans_G[i*ww +index]=255;
			//else if(ans_G[i*ww +index]<0) ans_G[i*ww +index]=0;
		}
		// blue
		setup_col(row0, row1, ww, h, index, ans_B);
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_B[i*ww +index]=row0[3*i/2];
			else ans_B[i*ww +index]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[i*ww +index];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		for(int i=0; i<hh; ++i){
			ans_B[i*ww +index]=(int)((float)ans_B[i*ww +index]/B_rate);
			//if(ans_B[i*ww +index]>255) ans_B[i*ww +index]=255;
			//else if(ans_B[i*ww +index]<0) ans_B[i*ww +index]=0;
		}
	}
}

__global__ void run_row(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh){
	//int bid = blockIdx.x;
	//int tid = threadIdx.x;
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<h){
		//__shared__ int weight[4]; // R_ori, G_ori, B_ori, e_aft
		//__shared__ float rate[3]; // R_rate, G_rate, B_rate
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		int index=(round+tid)*w;
		
		for(int i=0; i<w; ++i){ // compute weight
			R_ori+=tex1Dfetch(TR, index +i);
			G_ori+=tex1Dfetch(TG, index +i);
			B_ori+=tex1Dfetch(TB, index +i);
		}

		int row0[1280];
		int row1[1280];
		// red
		setup_row(row0, row1, w, round+tid, TR);
		e_aft=0;
		index=(round+tid)*w*2/3;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_R[index +i]=row0[3*i/2];
			else ans_R[index +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_R[index +i];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		for(int i=0; i<w*2/3; ++i){
			ans_R[index +i]=(int)((float)ans_R[index +i]/R_rate);
			//if(ans_B[index +i]>255) ans_B[index +i]=255;
			//else if(ans_B[index +i]<0) ans_B[index +i]=0;
		}
		// green
		/*
		setup_row(row0, row1, w, round+bid, TG, tid);
		__syncthreads();
		index=(round+bid)*w*2/3;
		if(tid<w*2/3){
			if(tid%2==0) ans_G[index +tid]=row[3*tid/2];
			else ans_G[index +tid]=row[w+ 3*(tid-1)/2 +2];
			__syncthreads();
			if(tid==0){
				weight[3]=0;
				for(int i=0; i<w*2/3; ++i)
					weight[3]+=ans_G[index+i];
				rate[1]=(float)weight[3]/((float)weight[1]*2.0/3.0);
			}
			__syncthreads();
			ans_G[index +tid]=(int)((float)ans_G[index +tid]/rate[1]);
		}
		__syncthreads();*/
		
		setup_row(row0, row1, w, round+tid, TG);
		e_aft=0;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_G[index +i]=row0[3*i/2];
			else ans_G[index +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_G[index +i];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		for(int i=0; i<w*2/3; ++i){
			ans_G[index +i]=(int)((float)ans_G[index +i]/G_rate);
		}
		
		// blue
		/*
		setup_row(row0, row1, w, round+bid, TB, tid);
		__syncthreads();
		index=(round+bid)*w*2/3;
		if(tid<w*2/3){
			if(tid%2==0) ans_B[index +tid]=row[3*tid/2];
			else ans_B[index +tid]=row[w+ 3*(tid-1)/2 +2];
			__syncthreads();
			if(tid==0){
				weight[3]=0;
				for(int i=0; i<w*2/3; ++i)
					weight[3]+=ans_B[index+i];
				rate[2]=(float)weight[3]/((float)weight[2]*2.0/3.0);
			}
			__syncthreads();
			ans_B[index +tid]=(int)((float)ans_B[index +tid]/rate[2]);
		}
		__syncthreads();*/
		
		setup_row(row0, row1, w, round+tid, TB);
		e_aft=0;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_B[index +i]=row0[3*i/2];
			else ans_B[index +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[index +i];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		for(int i=0; i<w*2/3; ++i){
			ans_B[index +i]=(int)((float)ans_B[index +i]/B_rate);
			//if(ans_B[index +i]>255) ans_B[index +i]=255;
			//else if(ans_B[index +i]<0) ans_B[index +i]=0;
		}
		
	}
}

void SR_kernel_down(int *ori_R, int *ori_G, int *ori_B, int *aft_R, int *aft_G, int *aft_B, int w, int h){
	float d0[3]={-0.022, 0.974, 0.227};
	float d1[3]={0.227, 0.974, -0.022};

	int *R, *G, *B;
	int *ans_R, *ans_G, *ans_B;
	int ww=w*2/3;
	int hh=h*2/3;

	cudaMalloc((void**)&R, w*h*sizeof(int));
	cudaMalloc((void**)&G, w*h*sizeof(int));
	cudaMalloc((void**)&B, w*h*sizeof(int));
	cudaMalloc((void**)&ans_R, w*h*sizeof(int)*2/3);
	cudaMalloc((void**)&ans_G, w*h*sizeof(int)*2/3);
	cudaMalloc((void**)&ans_B, w*h*sizeof(int)*2/3);
	
	cudaMemcpy(R, ori_R, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(G, ori_G, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B, ori_B, w*h*sizeof(int), cudaMemcpyHostToDevice);

	cudaBindTexture(0, TR, R);
	cudaBindTexture(0, TG, G);
	cudaBindTexture(0, TB, B);
	set_filter(d0, d1);

	int threads=64;
	int blocks=64;
	/* i want each block do a row, and each thread in a block handle a pixel */
	for(int i=0; i<(h-1)/(threads*blocks) +1; ++i)
		run_row<<<blocks, threads>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);
	
	for(int i=0; i<(ww-1)/(threads*blocks) +1; ++i)
		run_col<<<blocks, threads>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);
	

	cudaMemcpy(aft_R, ans_R, w*h*sizeof(int)*4/9, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_G, ans_G, w*h*sizeof(int)*4/9, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_B, ans_B, w*h*sizeof(int)*4/9, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(TR);
	cudaUnbindTexture(TG);
	cudaUnbindTexture(TB);
	cudaFree(R);
	cudaFree(G);
	cudaFree(B);
	cudaFree(ans_R);
	cudaFree(ans_G);
	cudaFree(ans_B);
}
