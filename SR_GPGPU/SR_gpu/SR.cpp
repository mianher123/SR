#include "all_lib.h"

using namespace cv;
int main(void){
	char *ImageName = "C:\\Users\\dada\\Desktop\\GPU_Image\\Koala.jpg";
	//char *ImageName = "C:\\Users\\dada\\Desktop\\GPU_Image\\Flower.jpg";
	IplImage *pImg;
	IplImage *pImg2;
	pImg=cvLoadImage(ImageName, 1);
	cvNamedWindow("ShowImage", 1); // create window
	//cvShowImage("ShowImage", pImg); // show image
	clock_t start, end, Up, find, Down;
	int w=pImg->width;
	int h=pImg->height;
	int bpp=pImg->nChannels;
	int ww=w*3/2;
	int hh=h*3/2;
	
	int *ori_R=(int*)malloc(sizeof(int)*w*h);
	int *ori_G=(int*)malloc(sizeof(int)*w*h);
	int *ori_B=(int*)malloc(sizeof(int)*w*h);
	
	int *DownUp_R=(int*)malloc(sizeof(int)*w*h);
	int *DownUp_G=(int*)malloc(sizeof(int)*w*h);
	int *DownUp_B=(int*)malloc(sizeof(int)*w*h);

	int *H_R=(int*)malloc(sizeof(int)*w*h);
	int *H_G=(int*)malloc(sizeof(int)*w*h);
	int *H_B=(int*)malloc(sizeof(int)*w*h);
	
	int *aft_R=(int*)malloc(sizeof(int)*ww*hh);
	int *aft_G=(int*)malloc(sizeof(int)*ww*hh);
	int *aft_B=(int*)malloc(sizeof(int)*ww*hh);
	
	int *ans_R=(int*)malloc(sizeof(int)*ww*hh);
	int *ans_G=(int*)malloc(sizeof(int)*ww*hh);
	int *ans_B=(int*)malloc(sizeof(int)*ww*hh);

	//SR_kernel_start(w, h, ww, hh);
	
	int img_size=w*h;
	int aft_size=ww*hh;
		for(int j=0; j<h; ++j){ // copy the original RGB
			for(int i=0; i<w; ++i){
				ori_R[j*w +i]=(unsigned char)pImg->imageData[(j*w +i)*bpp +2];
				ori_G[j*w +i]=(unsigned char)pImg->imageData[(j*w +i)*bpp +1];
				ori_B[j*w +i]=(unsigned char)pImg->imageData[(j*w +i)*bpp +0];
			}
		}

	start=clock();
	/******* run 1.5x down sample *******/
	//down(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h);
	SR_kernel_down(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h);
	Down=clock();
	SR_kernel_up(aft_R, aft_G, aft_B, DownUp_R, DownUp_G, DownUp_B, w*2/3, h*2/3);
	//up(aft_R, aft_G, aft_B, DownUp_R, DownUp_G, DownUp_B, w*2/3, h*2/3, w, h);
	
	/******* run 1.5x upsample *******/
	Up=clock();
	//up(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h, ww, hh);
	SR_kernel_up(ori_R, ori_G, ori_B, aft_R, aft_G, aft_B, w, h);
	

	/*******************************************************
		ori_R store original image
		aft_R store 1.5x upsample image
		DownUp_R store down sample then upsample image
	*******************************************************/
	
	for(int j=0; j<h; ++j){
		for(int i=0; i<w; ++i){
			H_R[j*w +i]=ori_R[j*w +i]-DownUp_R[j*w +i];
			H_G[j*w +i]=ori_G[j*w +i]-DownUp_G[j*w +i];
			H_B[j*w +i]=ori_B[j*w +i]-DownUp_B[j*w +i];
		}
	}
	
	find=clock();
	SR_kernel_find_neighbor(aft_R, aft_G, aft_B,
							DownUp_R, DownUp_G, DownUp_B,
							H_R, H_G, H_B,
							ans_R, ans_G, ans_B,
							w, h, ww, hh);
	end=clock();

	std::cout << std::dec << "Down time: " << 1000.0*(double)(Down-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "DownUp time: " << 1000.0*(double)(Up-Down)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "up time: " << 1000.0*(double)(find-Up)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "find neighbor time: " << 1000.0*(double)(end-find)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	std::cout << std::dec << "process time: " << 1000.0*(double)(end-start)/(double)CLOCKS_PER_SEC << " ms" << std::endl;
	//printf("clock per sec=%d\n", CLOCKS_PER_SEC);
	pImg2=cvCreateImage(cvSize(ww, hh), pImg->depth, pImg->nChannels);
	
	for(int j=0; j<hh; ++j){ // copy the ans to pImg2
		for(int i=0; i<ww; ++i){
			
			aft_R[j*ww +i]+=ans_R[j*ww +i];
			aft_G[j*ww +i]+=ans_G[j*ww +i];
			aft_B[j*ww +i]+=ans_B[j*ww +i];
			
			if( aft_R[j*ww +i]>255 ) aft_R[j*ww +i]=255;
			else if( aft_R[j*ww +i]<0 ) aft_R[j*ww +i]=0;
			if( aft_G[j*ww +i]>255 ) aft_G[j*ww +i]=255;
			else if( aft_G[j*ww +i]<0 ) aft_G[j*ww +i]=0;
			if( aft_B[j*ww +i]>255 ) aft_B[j*ww +i]=255;
			else if( aft_B[j*ww +i]<0 ) aft_B[j*ww +i]=0;
			
			pImg2->imageData[(j*ww +i)*bpp +2]=aft_R[j*ww +i];
			pImg2->imageData[(j*ww +i)*bpp +1]=aft_G[j*ww +i];
			pImg2->imageData[(j*ww +i)*bpp +0]=aft_B[j*ww +i];
		}
	}
	
	//char *img_save="C:\\Users\\dada\\Desktop\\GPU_Image\\Koala_cpu.jpg";
	//cvSaveImage(img_save, pImg2);
	cvShowImage("ShowImage", pImg2);
	waitKey(0);

	cvDestroyWindow("ShowImage");
	cvReleaseImage(&pImg);
	cvReleaseImage(&pImg2);
	return 0;
}
