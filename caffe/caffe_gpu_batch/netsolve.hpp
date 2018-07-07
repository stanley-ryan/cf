#ifndef	_NETSOLVE_H
#define _NETSOLVE_H

#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include"random_fill.hpp"
#include"caffe_read_train_data.hpp"


//#define PRINT_FORWARD
//#define PRINT_BACKWARD

#define  TRAIN_MODE 	1	//0:deploy mode 1:classification mode   2:detection mode
#define BLOCK	1024
//#define SC_INCLUDE_FX
//#include "systemc.h"





using namespace std;
//typedef  sc_fixed<64,40>  float;
//typedef  half float;

#define MULT_THREAD		1

#define EPS_ 			1e-05f
#define MAX_IN		2

typedef enum _n_layer{
	layer_conv=3,
	layer_convxnor,
	layer_pool,
	layer_lrn,
	layer_Prelu,
	layer_batchnorm,
	layer_scale,
	layer_Relu,
	layer_Softmax,
	layer_Softmax_loss,
	layer_Center_loss,
	layer_ip,
	layer_Drop,
	layer_Eltwise,
	layer_Concat,
	layer_logreg_loss,
	layer_region2d_loss,
	layer_none,
}e_layer_list;




typedef enum _n_kernel_action{
	pool_max=1,
	pool_avg,
	pool_min,
	relu_maxzero,
	prelu_share_channel,
	eltw_sum,
	eltw_prod,
	eltw_max,
	lrn_across_chan,
	lrn_within_chan,
	act_none,
}e_ker_act;


typedef enum _n_para_ini_method{
	para_ini_loading=0,
	para_ini_const_zero,
	para_ini_gaussian,
	para_ini_uninorm,
	para_ini_xavier,
}e_para_ini;


typedef enum _n_para_diff_method{
	e_para_normal,
	e_para1_prune,
	e_para_unchange,
	e_para_bp,
	e_para_unbp,
}e_para_mtd;


typedef struct kernel_inf{
	int ker_size;
 	int stride;
	int pad;
	int dialation;
	e_ker_act act;
 }t_ker_info;



typedef struct caffe_train_inf{
	float*     	para1_diff;
	float*     	para2_diff;
	float*     	para1_diff_hist;
	float*     	para2_diff_hist;
	int 	     	para1_size=0;
	int 	     	para2_size=0;
	float 	     	para1_lrmul=1;
	float 	     	para2_lrmul=1;
	float 	     	para1_lrdecay=1;
	float 	     	para2_lrdecay=1;
	e_para_ini	para1_ini_way=para_ini_loading;
	e_para_ini	para2_ini_way=para_ini_loading;
	e_para_mtd para_diff_way=e_para_normal;
	e_para_mtd para1_bp_mode=e_para_bp;
	e_para_mtd para2_bp_mode=e_para_bp;
	float 	prune_percent;
	float*    	para_diff_mask;
	int en_bias=1;
}t_train_inf;


typedef	struct layer_info{
	int layer_name;
	e_layer_list layer_tp;
	int output_num;
	int producer[MAX_IN];
	t_ker_info ker_nfo;
	float* para1_entry;
	float* para2_entry;
	float* input_buf[MAX_IN];
	float* output_buf;
	int in_dim[MAX_IN][4];	//C,H,W,BATCH
	int out_dim[4];	//C,H,W,BATCH
	t_train_inf train_inf;
}t_layer_info;


typedef	enum _lr_modle{			//learn rate change model
	lr_fixed=0,
	lr_step,
	lr_exp,
	lr_inv,
}e_lr;


#define DEFAULT_TEST_ITER_NUM 20

typedef	struct _train_strat{
	e_lr lr_mode=lr_fixed;
	float base_lr=0.001;
	float gamma=0.0001;
	float moment=0.9;
	float power=0.75;
	float decay=0.0005;
	int setp=1;
	int batch_size=1;
	int iter_size=64;
	int test_size=DEFAULT_TEST_ITER_NUM;
	int max_iter=999999;
	float target_loss;
	float target_accuracy=0.99;
	int loss_layer[10];
	int loss_layer_num=1;
	int is_shuffle=1;
	int is_withload=1;
	int record_itersize=DEFAULT_TEST_ITER_NUM;
}t_train_strat;


extern	float ini_data;

extern	float* convert_fp32_data_to_cftype(float* fp32_data, int data_size);
extern	float* convert_layer_outbuf_in_fp32(t_layer_info*  net_layers,int layer_id);
extern	int 	write_matrix(float*data,int D,int C,int H,int W,const char* filename);
extern	int 	write_matrix_longdata(long*data,int D,int C,int H,int W,const char* filename);
extern	int 	write_matrix_fix(float* data,int D,int C,int H,int W,const char* filename);
extern	int 	write_train_log(float loss,float accuarcy, int rnd,const char* filename);
extern	int	 write_gpu_matrix(float*gpu_data,int D,int C,int H,int W,const char* filename);
extern 	void reset_caffe_buffer(t_layer_info*  net_layers);
extern	int  initial_train_para(t_layer_info*  net_layers,int batch_num, int im_ch, int im_h, int im_w);
extern	void  convert_layer_outbuf_in_fp32_extern(t_layer_info*  net_layers,int layer_id,float* ext_buf);

#if (TRAIN_MODE==1)			//classification mode
extern	void  netbacksolve(t_layer_info*  net_layers, int top_layerid, int batch_num,  t_label* label_list);
#elif (TRAIN_MODE==2)		//detection mode
extern	void  netbacksolve(t_layer_info*  net_layers, int top_layerid, int batch_num,  t_box_list* box_list);
#endif


#if (TRAIN_MODE==1)			//classification mode
extern	void  netforwardsolve(t_layer_info*  net_layers, float**  im_data, int batch_num, int im_ch, int im_h, int im_w, t_label* label_list);
#elif (TRAIN_MODE==2)		//detection mode
extern	void  netforwardsolve(t_layer_info*  net_layers, float** im_data, int batch_num, int im_ch, int im_h, int im_w, t_box_list* box_list, string im_file[]);
#else						//deploy mode
extern	void  netforwardsolve(t_layer_info*  net_layers, float** im_data, int batch_num, int im_ch, int im_h, int im_w);
#endif
extern	void  net_update(t_layer_info*  net_layers, t_train_strat* train_config, int iter);
extern	void  print_matrix(char* dataname, float* data, int ch, int h, int w);
extern	void  net_record_weights(char* folder, t_layer_info*  net_layers);
extern	void print_gpu_data(float * Data, int datalen);
extern	void print_cpu_data(float * Data, int datalen);
extern	void print_stop_sign(char* sign);
extern	void 	gpu_xnor_nomalize_by_rows(float* input, int h, int w);
extern	void 	gpu_xnor_nomalize_by_cols(float* input, int h, int w);
extern	int 		get_bottom_num(t_layer_info*  net_layers,int layer_id);
extern	void 	clear_out_buffer(t_layer_info*  net_layers,int layer_id);


#define	MEASURE_TIME_VARIABLES()	\
	struct timeval			tv1, tv2;	\
	int				us

#define MEASURE_TIME_START(s) \
	gettimeofday(&tv1, NULL);	\
	printf("%20s--->%u:%u\n", s,tv1.tv_sec, tv1.tv_usec);

#define MEASURE_TIME_END(s)	\
	gettimeofday(&tv2, NULL);	\
	us = 1000000 * (tv2.tv_sec - tv1.tv_sec) + (tv2.tv_usec - tv1.tv_usec);	\
	printf("%20s %.4f s\n", s, (float)us/1000000)


#define  create_record_folder(train_tag,  iter)	\
{													\
	sprintf(folder,"./%s/round_%d/",train_tag,iter );	\
	sprintf(cmd, "mkdir -p %s",folder);	\
	system(cmd);		\
}


#endif
