#include "all_lib.h"

using namespace cv;

void up(IplImage *, IplImage *);
void copy(IplImage *, IplImage *);

int main(void){
	char *ImageName = "C:\\Users\\dada\\Documents\\Visual Studio 2010\\Projects\\SR_gpu\\Koala.jpg";
	IplImage *pImg;
	IplImage *pImg2;
	pImg=cvLoadImage(ImageName, 1);
	cvNamedWindow("ShowImage", 1); // create window
	//cvShowImage("ShowImage", pImg); // show image

	/* // original size
	pImg2=cvCreateImage(cvGetSize(pImg), pImg->depth, pImg->nChannels);
	//cvCopy(pImg, pImg2, NULL);
	copy(pImg, pImg2);
	*/
	clock_t start, end;
	int w=pImg->width;
	int h=pImg->height;
	int bpp=pImg->nChannels;
	int ww=w*3/2;
	int hh=h*3/2;
	
	int *ori_R=(int*)malloc(sizeof(int)*w*h);
	int *ori_G=(int*)malloc(sizeof(int)*w*h);
	int *ori_B=(int*)malloc(sizeof(int)*w*h);
	int *aft_R=(int*)malloc(sizeof(int)*w*h*9/4);
	int *aft_G=(int*)malloc(sizeof(int)*w*h*9/4);
	int *aft_B=(int*)malloc(sizeof(int)*w*h*9/4);

	for(int j=0; j<pImg->height; ++j){ // copy the original RGB
		for(int i=0; i<pImg->width; ++i){
			ori_R[j*w +i]=(unsigned char)pImg->imageData[(j*w +i)*bpp +2];
			ori_G[j*w +i]=(unsigned char)pImg->imageData[(j*w +i)*bpp +1];
			ori_B[j*w +i]=(unsigned char)pImg->imageData[(j*w +i)*bpp +0];
		}
	}

	/*
	for(int j=0; j<pImg->height; ++j){ // test ori ok or not
		for(int i=0; i<pImg->width; ++i){
			pImg2->imageData[(j*w +i)*bpp +2]=ori_R[j*w +i];
			pImg2->imageData[(j*w +i)*bpp +1]=ori_G[j*w +i];
			pImg2->imageData[(j*w +i)*bpp +0]=ori_B[j*w +i];
		}
	}
	*/

	start=clock();
	SR_kernel(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h); // launch kernel
	end=clock();
	std::cout << std::dec << "process time: " << 1000.0*(double)(end-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	
	pImg2=cvCreateImage(cvSize(pImg->width*3/2, pImg->height*3/2), pImg->depth, pImg->nChannels); // upsample
	
	for(int j=0; j<pImg->height*3/2; ++j){ // copy the ans to pImg2
		for(int i=0; i<pImg->width*3/2; ++i){
			pImg2->imageData[(j*ww +i)*bpp +2]=aft_R[j*ww +i];
			pImg2->imageData[(j*ww +i)*bpp +1]=aft_G[j*ww +i];
			pImg2->imageData[(j*ww +i)*bpp +0]=aft_B[j*ww +i];
		}
	}
	
	//up(pImg, pImg2);
	cvShowImage("ShowImage", pImg2);

	waitKey(0);

	cvDestroyWindow("ShowImage");
	cvReleaseImage(&pImg);
	cvReleaseImage(&pImg2);
	return 0;
}

void setup_col(int *row0, int *row1, IplImage *img, int h, int ww, int hh, int bpp, int index, int rgb){
	for(int i=0; i<hh; i++){
		if(i%3==0) row0[i]=(int)((unsigned char)img->imageData[((i*2/3)*ww+index)*bpp +rgb]);
		else row0[i]=0;

		if(i%3==2) row1[i]=(int)((unsigned char)img->imageData[(((i-2)*2/3+1)*ww+index)*bpp +rgb]);
		else row1[i]=0;
	}
}

void setup_row(int *row0, int *row1, IplImage *Img_ori, int ww, int bpp, int j, int rgb){
	int w=Img_ori->width;
	
	for(int i=0; i<ww; i++){ // setup row0 and row1
		if(i%3==0) row0[i]=(int)((unsigned char)Img_ori->imageData[(j*w+i*2/3)*bpp +rgb]);
		else row0[i]=0;
		
		if(i%3==2) row1[i]=(int)((unsigned char)Img_ori->imageData[(j*w+(i-2)*2/3+1)*bpp +rgb]);
		else row1[i]=0;

		if(row0[i]<0 || row1[i]<0){ // just for testing
			std::cout << "error ";
		}
	}
	
}

int convolusion_col(double u0[5], double u1[5], int *row0, int *row1, int ww, int hh, int bpp, int index, IplImage *img, int rgb){
	int e_aft=0;
	int temp[2];
	for(int i=0; i<hh; i++){
		// convolusion
		if(i==0){
			temp[0]=(int)(u0[2]*row0[0] + u0[3]*row0[1] + u0[4]*row0[2]);
			temp[1]=(int)(u1[2]*row1[0] + u1[3]*row1[1] + u1[4]*row1[2]);
		}
		else if(i==1){
			temp[0]=(int)(u0[1]*row0[0] + u0[2]*row0[1] + u0[3]*row0[2] + u0[4]*row0[3]);
			temp[1]=(int)(u1[1]*row1[0] + u1[2]*row1[1] + u1[3]*row1[2] + u1[4]*row1[3]);
		}
		else if(i==hh-2){
			temp[0]=(int)(u0[0]*row0[hh-4] + u0[1]*row0[hh-3] + u0[2]*row0[hh-2] + u0[3]*row0[hh-1]);
			temp[1]=(int)(u1[0]*row1[hh-4] + u1[1]*row1[hh-3] + u1[2]*row1[hh-2] + u1[3]*row1[hh-1]);
		}
		else if (i==hh-1){
			temp[0]=(int)(u0[0]*row0[hh-3] + u0[1]*row0[hh-2] + u0[2]*row0[hh-1]);
			temp[1]=(int)(u1[0]*row1[hh-3] + u1[1]*row1[hh-2] + u1[2]*row1[hh-1]);
		}
		else{
			temp[0]=(int)(u0[0]*row0[i-2] + u0[1]*row0[i-1] + u0[2]*row0[i] + u0[3]*row0[i+1] + u0[4]*row0[i+2]);
			temp[1]=(int)(u1[0]*row1[i-2] + u1[1]*row1[i-1] + u1[2]*row1[i] + u1[3]*row1[i+1] + u1[4]*row1[i+2]);
		}
		// add result
		img->imageData[(i*ww+index)*bpp +rgb]=temp[0]+temp[1];
		e_aft+=(temp[0]+temp[1]);
	}

	return e_aft;
}

int convolusion(double u0[5], double u1[5], int *row0, int *row1, int ww, int bpp, int j, IplImage *Img_aft, int rgb){
	int e_aft=0;
	int temp[2];
	for(int i=0; i<ww; i++){ // ww=w*3/2
	// convolusion
		if(i==0){
			temp[0]=(int)(u0[2]*row0[0] + u0[3]*row0[1] + u0[4]*row0[2]);
			temp[1]=(int)(u1[2]*row1[0] + u1[3]*row1[1] + u1[4]*row1[2]);
		}
		else if(i==1){
			temp[0]=(int)(u0[1]*row0[0] + u0[2]*row0[1] + u0[3]*row0[2] + u0[4]*row0[3]);
			temp[1]=(int)(u1[1]*row1[0] + u1[2]*row1[1] + u1[3]*row1[2] + u1[4]*row1[3]);
		}
		else if(i==ww-2){
			temp[0]=(int)(u0[0]*row0[ww-4] + u0[1]*row0[ww-3] + u0[2]*row0[ww-2] + u0[3]*row0[ww-1]);
			temp[1]=(int)(u1[0]*row1[ww-4] + u1[1]*row1[ww-3] + u1[2]*row1[ww-2] + u1[3]*row1[ww-1]);
		}
		else if (i==ww-1){
			temp[0]=(int)(u0[0]*row0[ww-3] + u0[1]*row0[ww-2] + u0[2]*row0[ww-1]);
			temp[1]=(int)(u1[0]*row1[ww-3] + u1[1]*row1[ww-2] + u1[2]*row1[ww-1]);
		}
		else{
			temp[0]=(int)(u0[0]*row0[i-2] + u0[1]*row0[i-1] + u0[2]*row0[i] + u0[3]*row0[i+1] + u0[4]*row0[i+2]);
			temp[1]=(int)(u1[0]*row1[i-2] + u1[1]*row1[i-1] + u1[2]*row1[i] + u1[3]*row1[i+1] + u1[4]*row1[i+2]);
		}
		// add result
		Img_aft->imageData[(j*ww+i)*bpp +rgb]=temp[0]+temp[1];
		e_aft+=(temp[0]+temp[1]);
	}

	return e_aft;
}

void up(IplImage *Img_ori, IplImage *Img_aft){
	double u1[5]={-0.1, 0.119, 0.927, 0.6, -0.047};
	double u0[5]={-0.047, 0.6, 0.927, 0.119, -0.1};
	double R_rate, G_rate, B_rate;
	int R_ori, G_ori, B_ori, e_aft;
	int w=Img_ori->width;
	int h=Img_ori->height;
	int bpp=Img_ori->nChannels;
	int ww=w*3/2;
	int hh=h*3/2;
	int *row0;
	int *row1;
	
	row0=(int*)malloc(sizeof(int)*ww);
	row1=(int*)malloc(sizeof(int)*ww);

	for(int j=0; j<h; j++){ // do how many time in row
		R_ori=0;
		G_ori=0;
		B_ori=0;
		for(int i=0; i<w; i++){ // for each row in Img_ori(width)
			R_ori+=(int)((unsigned char)Img_ori->imageData[(j*w+i)*bpp +2]); // compute weight
			G_ori+=(int)((unsigned char)Img_ori->imageData[(j*w+i)*bpp +1]);
			B_ori+=(int)((unsigned char)Img_ori->imageData[(j*w+i)*bpp +0]);
		}
		// red
		setup_row(row0, row1, Img_ori, ww, bpp, j, 2);
		e_aft=convolusion(u0, u1, row0, row1, ww, bpp, j, Img_aft, 2);
		R_rate=(double)e_aft/(double)(R_ori*3.0/2.0);
		// green
		setup_row(row0, row1, Img_ori, ww, bpp, j, 1);
		e_aft=convolusion(u0, u1, row0, row1, ww, bpp, j, Img_aft, 1);
		G_rate=(double)e_aft/(double)(G_ori*3.0/2.0);
		// blue
		setup_row(row0, row1, Img_ori, ww, bpp, j, 0);
		e_aft=convolusion(u0, u1, row0, row1, ww, bpp, j, Img_aft, 0);
		B_rate=(double)e_aft/(double)(B_ori*3.0/2.0);
		for(int i=0; i<ww; i++){
			Img_aft->imageData[(j*ww+i)*bpp +2]=(int)((double)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2])/R_rate);
			Img_aft->imageData[(j*ww+i)*bpp +1]=(int)((double)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1])/G_rate);
			Img_aft->imageData[(j*ww+i)*bpp +0]=(int)((double)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0])/B_rate);
			
			if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2])>255 ) Img_aft->imageData[(j*ww+i)*bpp +2]=255;
			else if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2])<0 ) Img_aft->imageData[(j*ww+i)*bpp +2]=0;
				//std::cout << std::dec << "(j, i)=(" << j << ", " << i << "), value=" << (int)Img_aft->imageData[(j*ww+i)*bpp +2] << std::endl;
			
			if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1])>255 ) Img_aft->imageData[(j*ww+i)*bpp +1]=255;
			else if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1])<0 ) Img_aft->imageData[(j*ww+i)*bpp +1]=0;

			if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0])>255 ) Img_aft->imageData[(j*ww+i)*bpp +0]=255;
			else if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0])<0 ) Img_aft->imageData[(j*ww+i)*bpp +0]=0;
			
		}
	}
	row0=(int*)malloc(sizeof(int)*hh);
	row1=(int*)malloc(sizeof(int)*hh);

	for(int i=0; i<ww; i++){ // do how many time in column
		R_ori=0;
		G_ori=0;
		B_ori=0;
		for(int j=0; j<h; j++){
			R_ori+=(int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2]); // compute weight
			G_ori+=(int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1]);
			B_ori+=(int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0]);
		}
		// red
		setup_col(row0, row1, Img_aft, h, ww, hh, bpp, i, 2);
		e_aft=convolusion_col(u0, u1, row0, row1, ww, hh, bpp, i, Img_aft, 2);
		R_rate=(double)e_aft/(double)(R_ori*3.0/2.0);
		// green
		setup_col(row0, row1, Img_aft, h, ww, hh, bpp, i, 1);
		e_aft=convolusion_col(u0, u1, row0, row1, ww, hh, bpp, i, Img_aft, 1);
		G_rate=(double)e_aft/(double)(G_ori*3.0/2.0);
		// blue
		setup_col(row0, row1, Img_aft, h, ww, hh, bpp, i, 0);
		e_aft=convolusion_col(u0, u1, row0, row1, ww, hh, bpp, i, Img_aft, 0);
		B_rate=(double)e_aft/(double)(B_ori*3.0/2.0);
		for(int j=0; j<hh; j++){
			Img_aft->imageData[(j*ww+i)*bpp +2]=(int)((double)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2])/R_rate);
			Img_aft->imageData[(j*ww+i)*bpp +1]=(int)((double)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1])/G_rate);
			Img_aft->imageData[(j*ww+i)*bpp +0]=(int)((double)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0])/B_rate);
			
			if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2])>(int)((unsigned char)255) ) Img_aft->imageData[(j*ww+i)*bpp +2]=(int)((unsigned char)255);
			else if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +2])<(int)((unsigned char)0) ) Img_aft->imageData[(j*ww+i)*bpp +2]=(int)((unsigned char)0);
				//std::cout << std::dec << "(j, i)=(" << j << ", " << i << "), value=" << (int)Img_aft->imageData[(j*ww+i)*bpp +2] << std::endl;
			
			if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1])>(int)((unsigned char)255) ) Img_aft->imageData[(j*ww+i)*bpp +1]=(int)((unsigned char)255);
			else if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +1])<(int)((unsigned char)0) ) Img_aft->imageData[(j*ww+i)*bpp +1]=(int)((unsigned char)0);

			if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0])>(int)((unsigned char)255) ) Img_aft->imageData[(j*ww+i)*bpp +0]=(int)((unsigned char)255);
			else if( (int)((unsigned char)Img_aft->imageData[(j*ww+i)*bpp +0])<(int)((unsigned char)0) ) Img_aft->imageData[(j*ww+i)*bpp +0]=(int)((unsigned char)0);
		}
	}
	
}

void copy(IplImage *Img_ori, IplImage *Img_aft){ // just for testing
	int w=Img_ori->width;
	int h=Img_ori->height;
	int bpp=Img_ori->nChannels;

	for(int j=0; j<h; j++){
		for(int i=0; i<w; i++){
			Img_aft->imageData[(j*w+i)*bpp +0]=Img_ori->imageData[(j*w+i)*bpp +0];
			Img_aft->imageData[(j*w+i)*bpp +1]=Img_ori->imageData[(j*w+i)*bpp +1];
			Img_aft->imageData[(j*w+i)*bpp +2]=Img_ori->imageData[(j*w+i)*bpp +2];
			if(j==0)
				std::cout << std::dec << (int)((unsigned char)Img_aft->imageData[(j*w+i)*bpp +2]) << " ";
		}
	}
}