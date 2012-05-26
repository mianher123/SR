
texture<int, 1, cudaReadModeElementType> TR;
texture<int, 1, cudaReadModeElementType> TG;
texture<int, 1, cudaReadModeElementType> TB;
texture<float, 1, cudaReadModeElementType> Td0;
texture<float, 1, cudaReadModeElementType> Td1;

__device__ void setup_col(int *row0, int *row1, int ww, int h, int index, int *ans){
	for(int i=0; i<h; ++i){
		if(i==0){
			row0[i]=(int)(tex1Dfetch(Td0, 1)*ans[0*ww +index]
						 +tex1Dfetch(Td0, 2)*ans[1*ww +index]);
			row1[i]=(int)(tex1Dfetch(Td1, 1)*ans[0*ww +index]
						 +tex1Dfetch(Td1, 2)*ans[1*ww +index]);
		}
		else if(i==h-1){
			row0[i]=(int)(tex1Dfetch(Td0, 0)*ans[(i-1)*ww +index]
						 +tex1Dfetch(Td0, 1)*ans[i*ww +index]);
			row1[i]=(int)(tex1Dfetch(Td1, 0)*ans[(i-1)*ww +index]
						 +tex1Dfetch(Td1, 1)*ans[i*ww +index]);
		}
		else{
			row0[i]=(int)(tex1Dfetch(Td0, 0)*ans[(i-1)*ww +index]
						 +tex1Dfetch(Td0, 1)*ans[i*ww +index]
						 +tex1Dfetch(Td0, 2)*ans[(i+1)*ww +index]);
			row1[i]=(int)(tex1Dfetch(Td1, 0)*ans[(i-1)*ww +index]
						 +tex1Dfetch(Td1, 1)*ans[i*ww +index]
						 +tex1Dfetch(Td1, 2)*ans[(i+1)*ww +index]);
		}
	}
}

__device__ void setup_row(int *row0, int *row1, int w, int index, texture<int, 1, cudaReadModeElementType> rgb){
	for(int i=0; i<w; ++i){
		if(i==0){
			row0[i]=(int)(tex1Dfetch(Td0, 1)*tex1Dfetch(rgb, index*w +0)
						 +tex1Dfetch(Td0, 2)*tex1Dfetch(rgb, index*w +1));
			row1[i]=(int)(tex1Dfetch(Td1, 1)*tex1Dfetch(rgb, index*w +0)
						 +tex1Dfetch(Td1, 2)*tex1Dfetch(rgb, index*w +1));
		}
		else if(i==w-1){
			row0[i]=(int)(tex1Dfetch(Td0, 0)*tex1Dfetch(rgb, index*w +i-1)
						 +tex1Dfetch(Td0, 1)*tex1Dfetch(rgb, index*w +i));
			row1[i]=(int)(tex1Dfetch(Td1, 0)*tex1Dfetch(rgb, index*w +i-1)
						 +tex1Dfetch(Td1, 1)*tex1Dfetch(rgb, index*w +i));
		}
		else{
			row0[i]=(int)(tex1Dfetch(Td0, 0)*tex1Dfetch(rgb, index*w +i-1)
						 +tex1Dfetch(Td0, 1)*tex1Dfetch(rgb, index*w +i)
						 +tex1Dfetch(Td0, 2)*tex1Dfetch(rgb, index*w +i+1));
			row1[i]=(int)(tex1Dfetch(Td1, 0)*tex1Dfetch(rgb, index*w +i-1)
						 +tex1Dfetch(Td1, 1)*tex1Dfetch(rgb, index*w +i)
						 +tex1Dfetch(Td1, 2)*tex1Dfetch(rgb, index*w +i+1));
		}
	}
}

__global__ void run_col(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<ww){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		for(int i=0; i<h; ++i){ // compute weight
			R_ori+=ans_R[i*ww +round+tid];
			G_ori+=ans_G[i*ww +round+tid];
			B_ori+=ans_B[i*ww +round+tid];
		}

		int row0[270];
		int row1[270];
		//red
		setup_col(row0, row1, ww, h, round+tid, ans_R);
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_R[i*ww +round+tid]=row0[3*i/2];
			else ans_R[i*ww +round+tid]=row1[3*(i-1)/2 +2];
			e_aft+=ans_R[i*ww +round+tid];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		for(int i=0; i<hh; ++i){
			ans_R[i*ww +round+tid]=(int)((float)ans_R[i*ww +round+tid]/R_rate);
			if(ans_R[i*ww +round+tid]>255) ans_R[i*ww +round+tid]=255;
			else if(ans_R[i*ww +round+tid]<0) ans_R[i*ww +round+tid]=0;
		}
		// green
		setup_col(row0, row1, ww, h, round+tid, ans_G);
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_G[i*ww +round+tid]=row0[3*i/2];
			else ans_G[i*ww +round+tid]=row1[3*(i-1)/2 +2];
			e_aft+=ans_G[i*ww +round+tid];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		for(int i=0; i<hh; ++i){
			ans_G[i*ww +round+tid]=(int)((float)ans_G[i*ww +round+tid]/G_rate);
			if(ans_G[i*ww +round+tid]>255) ans_G[i*ww +round+tid]=255;
			else if(ans_G[i*ww +round+tid]<0) ans_G[i*ww +round+tid]=0;
		}
		// blue
		setup_col(row0, row1, ww, h, round+tid, ans_B);
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_B[i*ww +round+tid]=row0[3*i/2];
			else ans_B[i*ww +round+tid]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[i*ww +round+tid];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		for(int i=0; i<hh; ++i){
			ans_B[i*ww +round+tid]=(int)((float)ans_B[i*ww +round+tid]/B_rate);
			if(ans_B[i*ww +round+tid]>255) ans_B[i*ww +round+tid]=255;
			else if(ans_B[i*ww +round+tid]<0) ans_B[i*ww +round+tid]=0;
		}
	}
}

__global__ void run_row(int round, int *ans_R, int *ans_G, int *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<h){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		for(int i=0; i<w; ++i){ // compute weight
			R_ori+=tex1Dfetch(TR, (round+tid)*w +i);
			G_ori+=tex1Dfetch(TG, (round+tid)*w +i);
			B_ori+=tex1Dfetch(TB, (round+tid)*w +i);
		}

		int row0[360];
		int row1[360];
		// red
		setup_row(row0, row1, w, round+tid, TR);
		e_aft=0;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_R[(round+tid)*w*2/3 +i]=row0[3*i/2];
			else ans_R[(round+tid)*w*2/3 +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_R[(round+tid)*w*2/3 +i];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		for(int i=0; i<w*2/3; ++i){
			ans_R[(round+tid)*w*2/3 +i]=(int)((float)ans_R[(round+tid)*w*2/3 +i]/R_rate);
			if(ans_R[(round+tid)*w*2/3 +i]>255) ans_R[(round+tid)*w*2/3 +i]=255;
			else if(ans_R[(round+tid)*w*2/3 +i]<0) ans_R[(round+tid)*w*2/3 +i]=0;
		}
		// green
		setup_row(row0, row1, w, round+tid, TG);
		e_aft=0;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_G[(round+tid)*w*2/3 +i]=row0[3*i/2];
			else ans_G[(round+tid)*w*2/3 +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_G[(round+tid)*w*2/3 +i];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		for(int i=0; i<w*2/3; ++i){
			ans_G[(round+tid)*w*2/3 +i]=(int)((float)ans_G[(round+tid)*w*2/3 +i]/G_rate);
			if(ans_G[(round+tid)*w*2/3 +i]>255) ans_G[(round+tid)*w*2/3 +i]=255;
			else if(ans_G[(round+tid)*w*2/3 +i]<0) ans_G[(round+tid)*w*2/3 +i]=0;
		}
		// blue
		setup_row(row0, row1, w, round+tid, TB);
		e_aft=0;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_B[(round+tid)*w*2/3 +i]=row0[3*i/2];
			else ans_B[(round+tid)*w*2/3 +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[(round+tid)*w*2/3 +i];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		for(int i=0; i<w*2/3; ++i){
			ans_B[(round+tid)*w*2/3 +i]=(int)((float)ans_B[(round+tid)*w*2/3 +i]/B_rate);
			if(ans_B[(round+tid)*w*2/3 +i]>255) ans_B[(round+tid)*w*2/3 +i]=255;
			else if(ans_B[(round+tid)*w*2/3 +i]<0) ans_B[(round+tid)*w*2/3 +i]=0;
		}
	}
}

void SR_kernel_down(int *ori_R, int *ori_G, int *ori_B, int *aft_R, int *aft_G, int *aft_B, int w, int h){
	float d0[3]={-0.022, 0.974, 0.227};
	float d1[3]={0.227, 0.974, -0.022};

	int *R, *G, *B;
	int *ans_R, *ans_G, *ans_B;
	int *d_d0, *d_d1;
	int ww=w*2/3;
	int hh=h*2/3;

	cudaMalloc((void**)&R, w*h*sizeof(int));
	cudaMalloc((void**)&G, w*h*sizeof(int));
	cudaMalloc((void**)&B, w*h*sizeof(int));
	cudaMalloc((void**)&ans_R, w*h*sizeof(int)*2/3);
	cudaMalloc((void**)&ans_G, w*h*sizeof(int)*2/3);
	cudaMalloc((void**)&ans_B, w*h*sizeof(int)*2/3);
	cudaMalloc((void**)&d_d0, 3*sizeof(float));
	cudaMalloc((void**)&d_d1, 3*sizeof(float));
	
	cudaMemcpy(R, ori_R, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(G, ori_G, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(B, ori_B, w*h*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d0, d0, 3*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_d1, d1, 3*sizeof(float), cudaMemcpyHostToDevice);

	cudaBindTexture(0, TR, R);
	cudaBindTexture(0, TG, G);
	cudaBindTexture(0, TB, B);
	cudaBindTexture(0, Td0, d_d0);
	cudaBindTexture(0, Td1, d_d1);

	int threads=64;
	int blocks=64;
	for(int i=0; i<(h-1)/(threads*blocks) +1; ++i)
		run_row<<<threads, blocks>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);

	for(int i=0; i<(ww-1)/(threads*blocks) +1; ++i)
		run_col<<<threads, blocks>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);
	

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
	cudaFree(d_d0);
	cudaFree(d_d1);
}
