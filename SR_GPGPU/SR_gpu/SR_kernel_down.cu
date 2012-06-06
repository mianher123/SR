
texture<unsigned char, 1, cudaReadModeElementType> TR;
texture<unsigned char, 1, cudaReadModeElementType> TG;
texture<unsigned char, 1, cudaReadModeElementType> TB;
texture<unsigned char ,1, cudaReadModeElementType> TansR;
texture<unsigned char ,1, cudaReadModeElementType> TansG;
texture<unsigned char ,1, cudaReadModeElementType> TansB;
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

__device__ unsigned char clamp(int value){
	if(value > 255) return (unsigned char)255;
	else if(value < 0) return (unsigned char)0;
	else return value;
}

__global__ void run_col(
	int col_base,
	unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B,
	int w, int h, int ww, int hh){


	// Calculate processing column number.
	int col_num = blockDim.x * blockIdx.x + threadIdx.x + col_base;

	if(col_num < ww){ 

		// Sum of pixels on this column in the original and the output image.
		int ori_sum, aft_sum;

		//float R_ratio, G_ratil, B_ratio;
		float norm_factor;

		// Index of base for this column.
		int col_ori_index_base = col_num;
		int col_aft_index_base = col_num;

		// Storage for current row offset in the original and the output image.
		int ori_index_offset, aft_index_offset;

		// Register storage for currently processing piece of original image.
		unsigned char ori_piece[4];

		// Temporary storage for pixel value.
		unsigned char temp0, temp1;

		// RED
		// Initialize offset and sum of column.
		ori_sum = 0; aft_sum = 0; ori_index_offset = 0; aft_index_offset = 0;

		// Apply filter for each group of two pixels.
		ori_piece[3] = tex1Dfetch(TansR, col_ori_index_base);
		#pragma unroll
		for(; aft_index_offset <= (hh-2)*ww; ori_index_offset += 3*ww, aft_index_offset += 2*ww){

			ori_piece[0] = ori_piece[3];
			ori_piece[1] = tex1Dfetch(TansR, col_ori_index_base + ori_index_offset);
			ori_piece[2] = tex1Dfetch(TansR, col_ori_index_base + ori_index_offset + ww);
			ori_piece[3] = tex1Dfetch(TansR, col_ori_index_base + ori_index_offset + 2*ww);
			ori_sum += (int)ori_piece[1] + (int)ori_piece[2] + (int)ori_piece[3];

			temp0 = clamp (
				d_d0[0]*(int)ori_piece[0] +
				d_d0[1]*(int)ori_piece[1] +
				d_d0[2]*(int)ori_piece[2] );

			temp1 = clamp(
				d_d1[0]*(int)ori_piece[1] +
				d_d1[1]*(int)ori_piece[2] +
				d_d1[2]*(int)ori_piece[3]);

			ans_R[col_aft_index_base + aft_index_offset] = (unsigned char)temp0;
			ans_R[col_aft_index_base + aft_index_offset + ww] = (unsigned char)temp1;
			aft_sum += (int)temp0 + (int)temp1;
		}

		// Normalization.
		norm_factor =  ((float)ori_sum*2.0/3.0) / (float)aft_sum;
		for(int i=0; i<hh; ++i){
			temp0 = clamp ( (int) ((float)ans_R[col_aft_index_base + i*ww] * norm_factor) );
			ans_R[col_aft_index_base + i*ww] = (unsigned char)temp0;
		}

		// GREEN
		// Initialize offset and sum of column.
		ori_sum = 0; aft_sum = 0; ori_index_offset = 0; aft_index_offset = 0;

		// Apply filter for each group of two pixels.
		ori_piece[3] = tex1Dfetch(TansG, col_ori_index_base);
		#pragma unroll
		for(; aft_index_offset <= (hh-2)*ww; ori_index_offset += 3*ww, aft_index_offset += 2*ww){

			ori_piece[0] = ori_piece[3];
			ori_piece[1] = tex1Dfetch(TansG, col_ori_index_base + ori_index_offset);
			ori_piece[2] = tex1Dfetch(TansG, col_ori_index_base + ori_index_offset + ww);
			ori_piece[3] = tex1Dfetch(TansG, col_ori_index_base + ori_index_offset + 2*ww);
			ori_sum += (int)ori_piece[1] + (int)ori_piece[2] + (int)ori_piece[3];

			temp0 = clamp (
				d_d0[0]*(int)ori_piece[0] +
				d_d0[1]*(int)ori_piece[1] +
				d_d0[2]*(int)ori_piece[2] );

			temp1 = clamp(
				d_d1[0]*(int)ori_piece[1] +
				d_d1[1]*(int)ori_piece[2] +
				d_d1[2]*(int)ori_piece[3]);

			ans_G[col_aft_index_base + aft_index_offset] = (unsigned char)temp0;
			ans_G[col_aft_index_base + aft_index_offset + ww] = (unsigned char)temp1;
			aft_sum += (int)temp0 + (int)temp1;
		}

		// Normalization.
		norm_factor =  ((float)ori_sum*2.0/3.0) / (float)aft_sum;
		for(int i=0; i<hh; ++i){
			temp0 = clamp ( (int) ((float)ans_G[col_aft_index_base + i*ww] * norm_factor) );
			ans_G[col_aft_index_base + i*ww] = (unsigned char)temp0;
		}
		
		// BLUE
		// Initialize offset and sum of column.
		ori_sum = 0; aft_sum = 0; ori_index_offset = 0; aft_index_offset = 0;

		// Apply filter for each group of two pixels.
		ori_piece[3] = tex1Dfetch(TansB, col_ori_index_base);
		#pragma unroll
		for(; aft_index_offset <= (hh-2)*ww; ori_index_offset += 3*ww, aft_index_offset += 2*ww){

			ori_piece[0] = ori_piece[3];
			ori_piece[1] = tex1Dfetch(TansB, col_ori_index_base + ori_index_offset);
			ori_piece[2] = tex1Dfetch(TansB, col_ori_index_base + ori_index_offset + ww);
			ori_piece[3] = tex1Dfetch(TansB, col_ori_index_base + ori_index_offset + 2*ww);
			ori_sum += (int)ori_piece[1] + (int)ori_piece[2] + (int)ori_piece[3];

			temp0 = clamp (
				d_d0[0]*(int)ori_piece[0] +
				d_d0[1]*(int)ori_piece[1] +
				d_d0[2]*(int)ori_piece[2] );

			temp1 = clamp(
				d_d1[0]*(int)ori_piece[1] +
				d_d1[1]*(int)ori_piece[2] +
				d_d1[2]*(int)ori_piece[3]);

			ans_B[col_aft_index_base + aft_index_offset] = (unsigned char)temp0;
			ans_B[col_aft_index_base + aft_index_offset + ww] = (unsigned char)temp1;
			aft_sum += (int)temp0 + (int)temp1;
		}

		// Normalization.
		norm_factor =  ((float)ori_sum*2.0/3.0) / (float)aft_sum;
		for(int i=0; i<hh; ++i){
			temp0 = clamp ( (int) ((float)ans_B[col_aft_index_base + i*ww] * norm_factor) );
			ans_B[col_aft_index_base + i*ww] = (unsigned char)temp0;
		}
	}
}


__global__ void run_row(
	int row_base,
	unsigned char *ans_R, unsigned char *ans_G, unsigned char *ans_B,
	int w, int h, int ww, int hh,
	unsigned char *temp_R, unsigned char *temp_G, unsigned char *temp_B){

	// Calculate processing row number.
	int row_num = blockDim.x * blockIdx.x + threadIdx.x + row_base;

	if(row_num < h){ 

		// Sum of pixels on this row in the original and the output image.
		int ori_sum, aft_sum;

		//float R_ratio, G_ratil, B_ratio;
		float norm_factor;

		// Index of base for this row.
		int row_ori_index_base = row_num*w;
		int row_aft_index_base = row_num*ww;

		// Storage for current column offset in the original and the output image.
		int ori_index_offset, aft_index_offset;

		// Register storage for currently processing piece of original image.
		unsigned char ori_piece[4];

		// Temporary storage for pixel value.
		unsigned char temp0, temp1;

		// RED
		// Initialize offset and sum of row.
		ori_sum = 0; aft_sum = 0; ori_index_offset = 0; aft_index_offset = 0;

		// Apply filter for each group of two pixels.
		ori_piece[3] = tex1Dfetch(TR, row_ori_index_base);
		#pragma unroll
		for(; aft_index_offset <= ww-2; ori_index_offset += 3, aft_index_offset += 2){

			ori_piece[0] = ori_piece[3];
			ori_piece[1] = tex1Dfetch(TR, row_ori_index_base + ori_index_offset);
			ori_piece[2] = tex1Dfetch(TR, row_ori_index_base + ori_index_offset +1);
			ori_piece[3] = tex1Dfetch(TR, row_ori_index_base + ori_index_offset +2);
			ori_sum += (int)ori_piece[1] + (int)ori_piece[2] + (int)ori_piece[3];

			temp0 = clamp (
				d_d0[0]*(int)ori_piece[0] +
				d_d0[1]*(int)ori_piece[1] +
				d_d0[2]*(int)ori_piece[2] );

			temp1 = clamp(
				d_d1[0]*(int)ori_piece[1] +
				d_d1[1]*(int)ori_piece[2] +
				d_d1[2]*(int)ori_piece[3] );

			ans_R[row_aft_index_base + aft_index_offset] = (unsigned char)temp0;
			ans_R[row_aft_index_base + aft_index_offset + 1] = (unsigned char)temp1;
			aft_sum += (int)temp0 + (int)temp1;
		}

		// Normalization.
		norm_factor =  ((float)ori_sum*2.0/3.0) / (float)aft_sum;
		for(int i=0; i<ww; ++i){
			temp0 = clamp ( (int) ((float)ans_R[row_aft_index_base +i] * norm_factor) );
			temp_R[row_aft_index_base + i]=ans_R[row_aft_index_base + i] = (unsigned char)temp0;
		}

		// GREEN
		// Initialize offset and sum of row.
		ori_sum = 0;
		aft_sum = 0;
		ori_index_offset = 0;
		aft_index_offset = 0;

		// Apply filter for each group of two pixels.
		ori_piece[3] = tex1Dfetch(TG, row_ori_index_base);
		#pragma unroll
		for(; aft_index_offset <= ww-2; ori_index_offset += 3, aft_index_offset += 2){

			ori_piece[0] = ori_piece[3];
			ori_piece[1] = tex1Dfetch(TG, row_ori_index_base + ori_index_offset);
			ori_piece[2] = tex1Dfetch(TG, row_ori_index_base + ori_index_offset +1);
			ori_piece[3] = tex1Dfetch(TG, row_ori_index_base + ori_index_offset +2);
			ori_sum += (int)ori_piece[1] + (int)ori_piece[2] + (int)ori_piece[3];

			temp0 = clamp (
				d_d0[0]*(int)ori_piece[0] +
				d_d0[1]*(int)ori_piece[1] +
				d_d0[2]*(int)ori_piece[2] );

			temp1 = clamp(
				d_d1[0]*(int)ori_piece[1] +
				d_d1[1]*(int)ori_piece[2] +
				d_d1[2]*(int)ori_piece[3] );

			ans_G[row_aft_index_base + aft_index_offset] = (unsigned char)temp0;
			ans_G[row_aft_index_base + aft_index_offset + 1] = (unsigned char)temp1;
			aft_sum += (int)temp0 + (int)temp1;
		}

		// Normalization.
		norm_factor =  ((float)ori_sum*2.0/3.0) / (float)aft_sum;
		for(int i=0; i<ww; ++i){
			temp0 = clamp ( (int) ((float)ans_G[row_aft_index_base +i] * norm_factor) );
			temp_G[row_aft_index_base + i] = ans_G[row_aft_index_base + i] = (unsigned char)temp0;
		}
		
		// BLUE
		// Initialize offset and sum of row.
		ori_sum = 0;
		aft_sum = 0;
		ori_index_offset = 0;
		aft_index_offset = 0;

		// Apply filter for each group of two pixels.
		ori_piece[3] = tex1Dfetch(TB, row_ori_index_base);
		#pragma unroll
		for(; aft_index_offset <= ww-2; ori_index_offset += 3, aft_index_offset += 2){

			ori_piece[0] = ori_piece[3];
			ori_piece[1] = tex1Dfetch(TB, row_ori_index_base + ori_index_offset);
			ori_piece[2] = tex1Dfetch(TB, row_ori_index_base + ori_index_offset +1);
			ori_piece[3] = tex1Dfetch(TB, row_ori_index_base + ori_index_offset +2);
			ori_sum += (int)ori_piece[1] + (int)ori_piece[2] + (int)ori_piece[3];

			temp0 = clamp (
				d_d0[0]*(int)ori_piece[0] +
				d_d0[1]*(int)ori_piece[1] +
				d_d0[2]*(int)ori_piece[2] );

			temp1 = clamp(
				d_d1[0]*(int)ori_piece[1] +
				d_d1[1]*(int)ori_piece[2] +
				d_d1[2]*(int)ori_piece[3] );

			ans_B[row_aft_index_base + aft_index_offset] = (unsigned char)temp0;
			ans_B[row_aft_index_base + aft_index_offset + 1] = (unsigned char)temp1;
			aft_sum += (int)temp0 + (int)temp1;
		}

		// Normalization.
		norm_factor =  ((float)ori_sum*2.0/3.0) / (float)aft_sum;
		for(int i=0; i<ww; ++i){
			temp0 = clamp ( (int) ((float)ans_B[row_aft_index_base +i] * norm_factor) );
			temp_B[row_aft_index_base + i] = ans_B[row_aft_index_base + i] = (unsigned char)temp0;
		}
	}
}

void SR_kernel_down(
	unsigned char *ori_R, unsigned char *ori_G, unsigned char *ori_B,
	unsigned char *aft_R, unsigned char *aft_G, unsigned char *aft_B,
	int w, int h){

	float d0[3]={0.227, 0.974, -0.022};
	float d1[3]={-0.022, 0.974, 0.227};

	unsigned char *R, *G, *B;
	unsigned char *ans_R, *ans_G, *ans_B;
	unsigned char *temp_R, *temp_G, *temp_B;
	int ww=w*2/3;
	int hh=h*2/3;

	cudaMalloc((void**)&R, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&G, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&B, w*h*sizeof(unsigned char));
	cudaMalloc((void**)&temp_R, ww*h*sizeof(unsigned char));
	cudaMalloc((void**)&temp_G, ww*h*sizeof(unsigned char));
	cudaMalloc((void**)&temp_B, ww*h*sizeof(unsigned char));
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
		run_row<<<blocks, threads>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh, temp_R, temp_G, temp_B);
	
	cudaBindTexture(0, TansR, temp_R);
	cudaBindTexture(0, TansG, temp_G);
	cudaBindTexture(0, TansB, temp_B);

	for(int i=0; i<(ww-1)/(threads*blocks) +1; ++i)
		run_col<<<blocks, threads>>>(i*threads*blocks, ans_R, ans_G, ans_B, w, h, ww, hh);
	

	cudaMemcpy(aft_R, ans_R, w*h*sizeof(unsigned char)*4/9, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_G, ans_G, w*h*sizeof(unsigned char)*4/9, cudaMemcpyDeviceToHost);
	cudaMemcpy(aft_B, ans_B, w*h*sizeof(unsigned char)*4/9, cudaMemcpyDeviceToHost);

	cudaUnbindTexture(TR);
	cudaUnbindTexture(TG);
	cudaUnbindTexture(TB);
	cudaUnbindTexture(TansR);
	cudaUnbindTexture(TansG);
	cudaUnbindTexture(TansB);
	cudaFree(R);
	cudaFree(G);
	cudaFree(B);
	cudaFree(ans_R);
	cudaFree(ans_G);
	cudaFree(ans_B);
	cudaFree(temp_R);
	cudaFree(temp_G);
	cudaFree(temp_B);
}
