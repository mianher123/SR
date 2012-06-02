
texture<unsigned char, 1, cudaReadModeElementType> TR;
texture<unsigned char, 1, cudaReadModeElementType> TG;
texture<unsigned char, 1, cudaReadModeElementType> TB;
/*
texture<float, 1, cudaReadModeElementType> Td0;
texture<float, 1, cudaReadModeElementType> Td1;
*/
__constant__ float d_d0[3];
__constant__ float d_d1[3];

//extern __shared__ int row[];

extern "C" void set_filter(float *d0, float *d1){
	cudaMemcpyToSymbol(d_d0, d0, 3*sizeof(float));
	cudaMemcpyToSymbol(d_d1, d1, 3*sizeof(float));
}

__device__ void setup_col(unsigned char *row0, unsigned char *row1, int ww, int h, int index, unsigned char *ans){
	int temp;

	temp=(int)(d_d0[1]*((int)ans[index])+d_d0[2]*((int)ans[ww +index]));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row0[0]=(unsigned char)temp;

	temp=(int)(d_d1[1]*((int)ans[index])+d_d1[2]*((int)ans[ww +index]));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row1[0]=(unsigned char)temp;

	temp=(int)(d_d0[0]*((int)ans[(h-2)*ww +index])+d_d0[1]*((int)ans[(h-1)*ww +index]));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row0[h-1]=(unsigned char)temp;

	temp=(int)(d_d1[0]*((int)ans[(h-2)*ww +index])+d_d1[1]*((int)ans[(h-1)*ww +index]));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row1[h-1]=(unsigned char)temp;

	#pragma unroll
	for(int i=1; i<h-1; ++i){
		temp=(int)(d_d0[0]*((int)ans[(i-1)*ww +index])+d_d0[1]*((int)ans[i*ww +index])+d_d0[2]*((int)ans[(i+1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[i]=(unsigned char)temp;

		temp=(int)(d_d1[0]*((int)ans[(i-1)*ww +index])+d_d1[1]*((int)ans[i*ww +index])+d_d1[2]*((int)ans[(i+1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[i]=(unsigned char)temp;
	}
}

__device__ void setup_row(unsigned char *row0, unsigned char *row1, int w, int index, texture<unsigned char, 1, cudaReadModeElementType> rgb){
	int temp;
	temp=(int)(d_d0[1]*(int)tex1Dfetch(rgb, index*w +0)+d_d0[2]*(int)tex1Dfetch(rgb, index*w +1));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row0[0]=(unsigned char)temp;

	temp=(int)(d_d1[1]*(int)tex1Dfetch(rgb, index*w +0)+d_d1[2]*(int)tex1Dfetch(rgb, index*w +1));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row1[0]=(unsigned char)temp;

	temp=(int)(d_d0[0]*(int)tex1Dfetch(rgb, index*w +w-2)+d_d0[1]*(int)tex1Dfetch(rgb, index*w +w-1));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row0[w-1]=(unsigned char)temp;

	temp=(int)(d_d1[0]*(int)tex1Dfetch(rgb, index*w +w-2)+d_d1[1]*(int)tex1Dfetch(rgb, index*w +w-1));
	if(temp>255) temp=255;
	else if(temp<0) temp=0;
	row1[w-1]=(unsigned char)temp;

	#pragma unroll
	for(int i=1; i<w-1; ++i){
		temp=(int)(d_d0[0]*(int)tex1Dfetch(rgb, index*w +i-1)+d_d0[1]*(int)tex1Dfetch(rgb, index*w +i)+d_d0[2]*(int)tex1Dfetch(rgb, index*w +i+1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[i]=(unsigned char)temp;

		temp=(int)(d_d1[0]*(int)tex1Dfetch(rgb, index*w +i-1)+d_d1[1]*(int)tex1Dfetch(rgb, index*w +i)+d_d1[2]*(int)tex1Dfetch(rgb, index*w +i+1));	
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[i]=(unsigned char)temp;
	}
}

__global__ void run_col(int round, unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B, int w, int h, int ww, int hh){
	int tid = blockDim.x * blockIdx.x + threadIdx.x;
	if(round+tid<ww){
		int R_ori=0, G_ori=0, B_ori=0; // store weight of original img
		int e_aft;
		float R_rate, G_rate, B_rate;
		int index=round+tid;
		for(int i=0; i<h; ++i){ // compute weight
			R_ori+=(int)ans_R[i*ww +index];
			G_ori+=(int)ans_G[i*ww +index];
			B_ori+=(int)ans_B[i*ww +index];
		}

		unsigned char row0[720];
		unsigned char row1[720];
		/////////////////////////////// red ////////////////////////////////////
		//setup_col(row0, row1, ww, h, index, ans_R);
		int temp;
		temp=(int)(d_d0[1]*((int)ans_R[index])+d_d0[2]*((int)ans_R[ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[0]=(unsigned char)temp;

		temp=(int)(d_d1[1]*((int)ans_R[index])+d_d1[2]*((int)ans_R[ww +index]));	
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[0]=(unsigned char)temp;

		temp=(int)(d_d0[0]*((int)ans_R[(h-2)*ww +index])+d_d0[1]*((int)ans_R[(h-1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[h-1]=(unsigned char)temp;

		temp=(int)(d_d1[0]*((int)ans_R[(h-2)*ww +index])+d_d1[1]*((int)ans_R[(h-1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[h-1]=(unsigned char)temp;

		#pragma unroll
		for(int i=1; i<h-1; ++i){
			temp=(int)(d_d0[0]*((int)ans_R[(i-1)*ww +index])+d_d0[1]*((int)ans_R[i*ww +index])+d_d0[2]*((int)ans_R[(i+1)*ww +index]));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row0[i]=(unsigned char)temp;
	
			temp=(int)(d_d1[0]*((int)ans_R[(i-1)*ww +index])+d_d1[1]*((int)ans_R[i*ww +index])+d_d1[2]*((int)ans_R[(i+1)*ww +index]));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row1[i]=(unsigned char)temp;
		}
		// setup_col() finish

		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_R[i*ww +index]=row0[3*i/2];
			else ans_R[i*ww +index]=row1[3*(i-1)/2 +2];
			e_aft+=ans_R[i*ww +index];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		if(R_rate<1.0){
			for(int i=0; i<hh; ++i){
				temp=(int)ans_R[i*ww +index];
				temp=(int)((float)temp/R_rate);
				if(temp>255) temp=255;
				else if(temp<0) temp=0;
				ans_R[i*ww +index]=(unsigned char)temp;
			}
		}
		else{
			for(int i=0; i<hh; ++i){
				temp=(int)ans_R[i*ww +index];
				temp=(int)((float)temp/R_rate);
				ans_R[i*ww +index]=(unsigned char)temp;
			}
		}
		////////////////////////// green ///////////////////////////
		//setup_col(row0, row1, ww, h, index, ans_G);
		temp=(int)(d_d0[1]*((int)ans_G[index])+d_d0[2]*((int)ans_G[ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[0]=(unsigned char)temp;

		temp=(int)(d_d1[1]*((int)ans_G[index])+d_d1[2]*((int)ans_G[ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[0]=(unsigned char)temp;

		temp=(int)(d_d0[0]*((int)ans_G[(h-2)*ww +index])+d_d0[1]*((int)ans_G[(h-1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[h-1]=(unsigned char)temp;
	
		temp=(int)(d_d1[0]*((int)ans_G[(h-2)*ww +index])+d_d1[1]*((int)ans_G[(h-1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[h-1]=(unsigned char)temp;

		#pragma unroll
		for(int i=1; i<h-1; ++i){
			temp=(int)(d_d0[0]*((int)ans_G[(i-1)*ww +index])+d_d0[1]*((int)ans_G[i*ww +index])+d_d0[2]*((int)ans_G[(i+1)*ww +index]));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row0[i]=(unsigned char)temp;

			temp=(int)(d_d1[0]*((int)ans_G[(i-1)*ww +index])+d_d1[1]*((int)ans_G[i*ww +index])+d_d1[2]*((int)ans_G[(i+1)*ww +index]));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row1[i]=(unsigned char)temp;
		}
		// setup_col() finish
		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_G[i*ww +index]=row0[3*i/2];
			else ans_G[i*ww +index]=row1[3*(i-1)/2 +2];
			e_aft+=ans_G[i*ww +index];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		if(G_rate<1.0){
			for(int i=0; i<hh; ++i){
				temp=(int)ans_G[i*ww +index];
				temp=(int)((float)temp/G_rate);
				if(temp>255) temp=255;
				else if(temp<0) temp=0;
				ans_G[i*ww +index]=(unsigned char)temp;
			}
		}
		else{
			for(int i=0; i<hh; ++i){
				temp=(int)ans_G[i*ww +index];
				temp=(int)((float)temp/G_rate);
				ans_G[i*ww +index]=(unsigned char)temp;
			}
		}
		///////////////////////////////// blue /////////////////////////////////
		//setup_col(row0, row1, ww, h, index, ans_B);
		temp=(int)(d_d0[1]*((int)ans_B[index])+d_d0[2]*((int)ans_B[ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[0]=(unsigned char)temp;

		temp=(int)(d_d1[1]*((int)ans_B[index])+d_d1[2]*((int)ans_B[ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[0]=(unsigned char)temp;

		temp=(int)(d_d0[0]*((int)ans_B[(h-2)*ww +index])+d_d0[1]*((int)ans_B[(h-1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[h-1]=(unsigned char)temp;

		temp=(int)(d_d1[0]*((int)ans_B[(h-2)*ww +index])+d_d1[1]*((int)ans_B[(h-1)*ww +index]));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[h-1]=(unsigned char)temp;

		#pragma unroll
		for(int i=1; i<h-1; ++i){
			temp=(int)(d_d0[0]*((int)ans_B[(i-1)*ww +index])+d_d0[1]*((int)ans_B[i*ww +index])+d_d0[2]*((int)ans_B[(i+1)*ww +index]));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row0[i]=(unsigned char)temp;
			
			temp=(int)(d_d1[0]*((int)ans_B[(i-1)*ww +index])+d_d1[1]*((int)ans_B[i*ww +index])+d_d1[2]*((int)ans_B[(i+1)*ww +index]));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row1[i]=(unsigned char)temp;
		}
		// setup_col() finish

		e_aft=0;
		for(int i=0; i<hh; ++i){
			if(i%2==0) ans_B[i*ww +index]=row0[3*i/2];
			else ans_B[i*ww +index]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[i*ww +index];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		if(B_rate<1.0){
			for(int i=0; i<hh; ++i){
				temp=(int)ans_B[i*ww +index];
				temp=(int)((float)temp/B_rate);
				if(temp>255) temp=255;
				else if(temp<0) temp=0;
				ans_B[i*ww +index]=(unsigned char)temp;
			}
		}
		else{
			for(int i=0; i<hh; ++i){
				temp=(int)ans_B[i*ww +index];
				temp=(int)((float)temp/B_rate);
				ans_B[i*ww +index]=(unsigned char)temp;
			}
		}
	}
}

__global__ void run_row(int round, unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B, int w, int h, int ww, int hh){
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
			R_ori+=(int)tex1Dfetch(TR, index +i);
			G_ori+=(int)tex1Dfetch(TG, index +i);
			B_ori+=(int)tex1Dfetch(TB, index +i);
		}

		unsigned char row0[1280];
		unsigned char row1[1280];
		index=round+tid;
		//////////////////////////////// red ////////////////////////////////////
		//setup_row(row0, row1, w, round+tid, TR);
		int temp;
		temp=(int)(d_d0[1]*(int)tex1Dfetch(TR, index*w +0)+d_d0[2]*(int)tex1Dfetch(TR, index*w +1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[0]=(unsigned char)temp;

		temp=(int)(d_d1[1]*(int)tex1Dfetch(TR, index*w +0)+d_d1[2]*(int)tex1Dfetch(TR, index*w +1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[0]=(unsigned char)temp;

		temp=(int)(d_d0[0]*(int)tex1Dfetch(TR, index*w +w-2)+d_d0[1]*(int)tex1Dfetch(TR, index*w +w-1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[w-1]=(unsigned char)temp;

		temp=(int)(d_d1[0]*(int)tex1Dfetch(TR, index*w +w-2)+d_d1[1]*(int)tex1Dfetch(TR, index*w +w-1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[w-1]=(unsigned char)temp;

		#pragma unroll
		for(int i=1; i<w-1; ++i){
			temp=(int)(d_d0[0]*(int)tex1Dfetch(TR, index*w +i-1)+d_d0[1]*(int)tex1Dfetch(TR, index*w +i)+d_d0[2]*(int)tex1Dfetch(TR, index*w +i+1));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row0[i]=(unsigned char)temp;
	
			temp=(int)(d_d1[0]*(int)tex1Dfetch(TR, index*w +i-1)+d_d1[1]*(int)tex1Dfetch(TR, index*w +i)+d_d1[2]*(int)tex1Dfetch(TR, index*w +i+1));	
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row1[i]=(unsigned char)temp;
		}
		// setup_row() finish


		e_aft=0;
		index=(round+tid)*ww;
		for(int i=0; i<ww; ++i){
			if(i%2==0) ans_R[index +i]=row0[3*i/2];
			else ans_R[index +i]=row1[3*(i-1)/2 +2];
			e_aft+=(int)ans_R[index +i];
		}
		R_rate=(float)e_aft/((float)R_ori*2.0/3.0);
		if(R_rate<1.0){
			for(int i=0; i<ww; ++i){
				temp=(int)ans_R[index +i];
				temp=(int)((float)temp/R_rate);
				if(temp>255) temp=255;
				else if(temp<0) temp=0;
				ans_R[index +i]=(unsigned char)temp;
			}
		}
		else{
			for(int i=0; i<ww; ++i){
				temp=(int)ans_R[index +i];
				temp=(int)((float)temp/R_rate);
				ans_R[index +i]=(unsigned char)temp;
			}
		}
		////////////////////// green //////////////////////////////
		//setup_row(row0, row1, w, round+tid, TG);
		index=round+tid;
		temp=(int)(d_d0[1]*(int)tex1Dfetch(TG, index*w +0)+d_d0[2]*(int)tex1Dfetch(TG, index*w +1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[0]=(unsigned char)temp;

		temp=(int)(d_d1[1]*(int)tex1Dfetch(TG, index*w +0)+d_d1[2]*(int)tex1Dfetch(TR, index*w +1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[0]=(unsigned char)temp;

		temp=(int)(d_d0[0]*(int)tex1Dfetch(TG, index*w +w-2)+d_d0[1]*(int)tex1Dfetch(TG, index*w +w-1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[w-1]=(unsigned char)temp;

		temp=(int)(d_d1[0]*(int)tex1Dfetch(TG, index*w +w-2)+d_d1[1]*(int)tex1Dfetch(TG, index*w +w-1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[w-1]=(unsigned char)temp;

		#pragma unroll
		for(int i=1; i<w-1; ++i){
			temp=(int)(d_d0[0]*(int)tex1Dfetch(TG, index*w +i-1)+d_d0[1]*(int)tex1Dfetch(TG, index*w +i)+d_d0[2]*(int)tex1Dfetch(TG, index*w +i+1));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row0[i]=(unsigned char)temp;
				
			temp=(int)(d_d1[0]*(int)tex1Dfetch(TG, index*w +i-1)+d_d1[1]*(int)tex1Dfetch(TG, index*w +i)+d_d1[2]*(int)tex1Dfetch(TG, index*w +i+1));	
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row1[i]=(unsigned char)temp;
		}
		// setup_row() finish

		e_aft=0;
		index=(round+tid)*w*2/3;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_G[index +i]=row0[3*i/2];
			else ans_G[index +i]=row1[3*(i-1)/2 +2];
			e_aft+=(int)ans_G[index +i];
		}
		G_rate=(float)e_aft/((float)G_ori*2.0/3.0);
		if(G_rate<1.0){
			for(int i=0; i<ww; ++i){
				temp=(int)ans_G[index +i];
				temp=(int)((float)temp/G_rate);
				if(temp>255) temp=255;
				else if(temp<0) temp=0;
				ans_G[index +i]=(unsigned char)temp;
			}
		}
		else{
			for(int i=0; i<ww; ++i){
				temp=(int)ans_G[index +i];
				temp=(int)((float)temp/G_rate);
				ans_G[index +i]=(unsigned char)temp;
			}
		}
		
		////////////////////////// blue ////////////////////////////
		
		//setup_row(row0, row1, w, round+tid, TB);
		index=round+tid;
		temp=(int)(d_d0[1]*(int)tex1Dfetch(TB, index*w +0)+d_d0[2]*(int)tex1Dfetch(TB, index*w +1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[0]=(unsigned char)temp;

		temp=(int)(d_d1[1]*(int)tex1Dfetch(TB, index*w +0)+d_d1[2]*(int)tex1Dfetch(TB, index*w +1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[0]=(unsigned char)temp;

		temp=(int)(d_d0[0]*(int)tex1Dfetch(TB, index*w +w-2)+d_d0[1]*(int)tex1Dfetch(TB, index*w +w-1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row0[w-1]=(unsigned char)temp;
	
		temp=(int)(d_d1[0]*(int)tex1Dfetch(TB, index*w +w-2)+d_d1[1]*(int)tex1Dfetch(TB, index*w +w-1));
		if(temp>255) temp=255;
		else if(temp<0) temp=0;
		row1[w-1]=(unsigned char)temp;
	
		#pragma unroll
		for(int i=1; i<w-1; ++i){
			temp=(int)(d_d0[0]*(int)tex1Dfetch(TB, index*w +i-1)+d_d0[1]*(int)tex1Dfetch(TB, index*w +i)+d_d0[2]*(int)tex1Dfetch(TB, index*w +i+1));
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row0[i]=(unsigned char)temp;
	
			temp=(int)(d_d1[0]*(int)tex1Dfetch(TB, index*w +i-1)+d_d1[1]*(int)tex1Dfetch(TB, index*w +i)+d_d1[2]*(int)tex1Dfetch(TB, index*w +i+1));	
			if(temp>255) temp=255;
			else if(temp<0) temp=0;
			row1[i]=(unsigned char)temp;
		}
		// setup_row() finish

		e_aft=0;
		index=(round+tid)*w*2/3;
		for(int i=0; i<w*2/3; ++i){
			if(i%2==0) ans_B[index +i]=row0[3*i/2];
			else ans_B[index +i]=row1[3*(i-1)/2 +2];
			e_aft+=ans_B[index +i];
		}
		B_rate=(float)e_aft/((float)B_ori*2.0/3.0);
		if(B_rate<1.0){
			for(int i=0; i<ww; ++i){
				temp=(int)ans_B[index +i];
				temp=(int)((float)temp/B_rate);
				if(temp>255) temp=255;
				else if(temp<0) temp=0;
				ans_B[index +i]=(unsigned char)temp;
			}
		}
		else{
			for(int i=0; i<ww; ++i){
				temp=(int)ans_B[index +i];
				temp=(int)((float)temp/B_rate);
				ans_B[index +i]=(unsigned char)temp;
			}
		}
		
	}
}

void SR_kernel_down(
	unsigned char *ori_R, unsigned char *ori_G, unsigned char *ori_B,
	unsigned char *aft_R, unsigned char *aft_G, unsigned char *aft_B,
	int w, int h){
	float d0[3]={-0.022, 0.974, 0.227};
	float d1[3]={0.227, 0.974, -0.022};

	unsigned char *R, *G, *B;
	unsigned char *ans_R, *ans_G, *ans_B;
	int ww=w*2/3;
	int hh=h*2/3;

	cudaMalloc((void**)&R, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&G, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&B, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&ans_R, w*h*sizeof(unsigned char)*2/3);
	cudaMalloc((void**)&ans_G, w*h*sizeof(unsigned char)*2/3);
	cudaMalloc((void**)&ans_B, w*h*sizeof(unsigned char)*2/3);
	
	cudaMemcpy(R, ori_R, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(G, ori_G, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);
	cudaMemcpy(B, ori_B, w*h*sizeof(unsigned char), cudaMemcpyHostToDevice);

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
	

	cudaMemcpy(aft_R, ans_R, w*h*sizeof(unsigned char)*4/9, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_G, ans_G, w*h*sizeof(unsigned char)*4/9, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_B, ans_B, w*h*sizeof(unsigned char)*4/9, cudaMemcpyDeviceToHost);

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
