#ifndef	_RANDOM_FILL_H
#define	_RANDOM_FILL_H

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <random>

extern	int gaussian_fill(float sigma,float mean,float* buf, int num);
extern	int unirom_fill(float lb, float ub, float* buf, int num);
extern	int ZO_fill(int lb, int ub, float* buf, int num);


extern	int rnd_int(int lb, int ub, int* buf, int num);
extern 	int rnd_mask(int mask_size, float* buf, float percent,float mask_data, float bg_data);

#endif
