

void SR_kernel_up(int *, int *, int *, int *, int *, int *, int, int, int, int);
void SR_kernel_down(int *, int *, int *, int *, int *, int *, int, int);
void SR_kernel_find_neighbor(int *,int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int *, int , int, int, int);
void SR_kernel_start(int, int, int, int);
void SR_kernel_end();
void down(int *ori_R, int *ori_G, int *ori_B,
	int *ans_R, int *ans_G, int *ans_B,
	int w, int h);
void up(int *ori_R, int *ori_G, int *ori_B, int *aft_R, int *aft_G, int *aft_B, int w, int h, int ww, int hh);
