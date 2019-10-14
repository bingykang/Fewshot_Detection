#include <TH/TH.h>
#include <THC/THC.h>
#include <stdio.h>

#define GPU 1

extern THCState *state;
/*
void mean_cpu(float *x, int batch, int filters, int spatial, float *mean)
{
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                mean[i] += x[index];
            }
        }
        mean[i] *= scale;
    }
}

void variance_cpu(float *x, float *mean, int batch, int filters, int spatial, float *variance)
{
    //float scale = 1./(batch * spatial - 1);
    float scale = 1./(batch * spatial);
    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance[i] += pow((x[index] - mean[i]), 2);
            }
        }
        variance[i] *= scale;
    }
}



void copy_cpu(int N, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] = X[i*INCX];
}
void axpy_cpu(int N, float ALPHA, float *X, int INCX, float *Y, int INCY)
{
    int i;
    for(i = 0; i < N; ++i) Y[i*INCY] += ALPHA*X[i*INCX];
}

void scal_cpu(int N, float ALPHA, float *X, int INCX)
{
    int i;
    for(i = 0; i < N; ++i) X[i*INCX] *= ALPHA;
}
void normalize_cpu(float *x, float *mean, float *variance, int batch, int filters, int spatial)
{
    int b, f, i;
    for(b = 0; b < batch; ++b){
        for(f = 0; f < filters; ++f){
            for(i = 0; i < spatial; ++i){
                int index = b*filters*spatial + f*spatial + i;
                x[index] = (x[index] - mean[f])/(sqrt(variance[f] + .00001f));
            }
        }
    }
}
*/
void add_bias(float *output, float *biases, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] += biases[i];
            }   
        }   
    }   
}

float sum_array(float *a, int n)
{
    int i;
    float sum = 0;
    for(i = 0; i < n; ++i) sum += a[i];
    return sum;
}
void scale_bias(float *output, float *scales, int batch, int n, int size)
{
    int i,j,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            for(j = 0; j < size; ++j){
                output[(b*n + i)*size + j] *= scales[i];
            }
        }
    }
}

void backward_bias(float *bias_updates, float *delta, int batch, int n, int size)
{
    int i,b;
    for(b = 0; b < batch; ++b){
        for(i = 0; i < n; ++i){
            bias_updates[i] += sum_array(delta+size*(i+b*n), size);
        }
    }
}


void backward_scale_cpu(float *x_norm, float *delta, int batch, int n, int size, float *scale_updates)
{
    int i,b,f;
    for(f = 0; f < n; ++f){
        float sum = 0;
        for(b = 0; b < batch; ++b){
            for(i = 0; i < size; ++i){
                int index = i + size*(f + n*b);
                sum += delta[index] * x_norm[index];
            }
        }
        scale_updates[f] += sum;
    }
}

void mean_delta_cpu(float *delta, float *variance, int batch, int filters, int spatial, float *mean_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        mean_delta[i] = 0;
        for (j = 0; j < batch; ++j) {
            for (k = 0; k < spatial; ++k) {
                int index = j*filters*spatial + i*spatial + k;
                mean_delta[i] += delta[index];
            }
        }
        mean_delta[i] *= (-1./sqrt(variance[i] + .00001f));
    }
}
void  variance_delta_cpu(float *x, float *delta, float *mean, float *variance, int batch, int filters, int spatial, float *variance_delta)
{

    int i,j,k;
    for(i = 0; i < filters; ++i){
        variance_delta[i] = 0;
        for(j = 0; j < batch; ++j){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + i*spatial + k;
                variance_delta[i] += delta[index]*(x[index] - mean[i]);
            }
        }
        variance_delta[i] *= -.5 * pow(variance[i] + .00001f, (float)(-3./2.));
    }
}
void normalize_delta_cpu(float *x, float *mean, float *variance, float *mean_delta, float *variance_delta, int batch, int filters, int spatial, float *delta)
{
    int f, j, k;
    for(j = 0; j < batch; ++j){
        for(f = 0; f < filters; ++f){
            for(k = 0; k < spatial; ++k){
                int index = j*filters*spatial + f*spatial + k;
                delta[index] = delta[index] * 1./(sqrt(variance[f] + .00001f)) + variance_delta[f] * 2. * (x[index] - mean[f]) / (spatial * batch) + mean_delta[f]/(spatial*batch);
            }
        }
    }
}

void bn_forward(THFloatTensor* in_t, THFloatTensor* x_t, THFloatTensor* x_norm_t, THFloatTensor* mean_t, THFloatTensor* rolling_mean_t, THFloatTensor* variance_t, THFloatTensor* rolling_variance_t, THFloatTensor* scales_t, THFloatTensor* biases_t, int train, THFloatTensor* out_t)
{
    float * input = THFloatTensor_data(in_t);
    float * output = THFloatTensor_data(out_t);
    float * x = THFloatTensor_data(x_t);
    float * x_norm = THFloatTensor_data(x_norm_t);
    float * mean = THFloatTensor_data(mean_t);
    float * rolling_mean = THFloatTensor_data(rolling_mean_t);
    float * variance = THFloatTensor_data(variance_t);
    float * rolling_variance = THFloatTensor_data(rolling_variance_t);
    float * scales = THFloatTensor_data(scales_t);
    float * biases = THFloatTensor_data(biases_t);
    
    THLongStorage* size_s = THFloatTensor_newSizeOf(out_t);
    long * size = THLongStorage_data(size_s);

    long batch = size[0]; 
    long out_c = size[1];
    long out_h = size[2];
    long out_w = size[3];

    copy_cpu(out_h*out_w*out_c*batch, input, 1, output, 1);
    
    if(train){
        mean_cpu(output, batch, out_c, out_h*out_w, mean);
        variance_cpu(output, mean, batch, out_c, out_h*out_w, variance);

        scal_cpu(out_c, .99, rolling_mean, 1);
        axpy_cpu(out_c, .01, mean, 1, rolling_mean, 1);
        scal_cpu(out_c, .99, rolling_variance, 1);
        axpy_cpu(out_c, .01, variance, 1, rolling_variance, 1);

        copy_cpu(out_c*out_h*out_w*batch, output, 1, x, 1);
        normalize_cpu(output, mean, variance, batch, out_c, out_h*out_w);   
        copy_cpu(out_c*out_h*out_w*batch, output, 1, x_norm, 1);
    } else {
        normalize_cpu(output, rolling_mean, rolling_variance, batch, out_c, out_h*out_w);
    }
    scale_bias(output, scales, batch, out_c, out_h*out_w);
    
    add_bias(output, biases, batch, out_c, out_h*out_w);
}

void bn_backward(THFloatTensor* grad_out_t, THFloatTensor* x_t, THFloatTensor* x_norm_t, THFloatTensor* mean_t, THFloatTensor* mean_delta_t, THFloatTensor* variance_t, THFloatTensor* variance_delta_t, THFloatTensor* scales_t,THFloatTensor* scale_delta_t, THFloatTensor* biases_t,THFloatTensor* bias_delta_t,  int train, THFloatTensor* grad_in_t)
{
    float * state_delta = THFloatTensor_data(grad_out_t);
    float * delta = THFloatTensor_data(grad_in_t);
    float * x = THFloatTensor_data(x_t);
    float * x_norm = THFloatTensor_data(x_norm_t);
    float * mean = THFloatTensor_data(mean_t);
    float * mean_delta = THFloatTensor_data(mean_delta_t);
    float * variance = THFloatTensor_data(variance_t);
    float * variance_delta = THFloatTensor_data(variance_delta_t);
    float * scales = THFloatTensor_data(scales_t);
    float * scale_delta = THFloatTensor_data(scale_delta_t);
    float * bias_delta = THFloatTensor_data(bias_delta_t);
   
    THLongStorage* size_s = THFloatTensor_newSizeOf(grad_out_t);
    long * size = THLongStorage_data(size_s);

    long batch = size[0]; 
    long out_c = size[1];
    long out_h = size[2];
    long out_w = size[3];
    
    
    copy_cpu(out_c*out_h*out_w*batch, state_delta, 1, delta, 1);
   
    backward_bias(bias_delta, delta, batch, out_c, out_h*out_w);

    backward_scale_cpu(x_norm, delta, batch, out_c, out_w*out_h, scale_delta);

    scale_bias(delta, scales, batch, out_c, out_h*out_w);

    mean_delta_cpu(delta, variance, batch, out_c, out_w*out_h, mean_delta);
    variance_delta_cpu(x, delta, mean, variance, batch, out_c, out_w*out_h, variance_delta);
    normalize_delta_cpu(x, mean, variance, mean_delta, variance_delta, batch, out_c, out_w*out_h, delta);

}

#ifdef GPU

//void bn_forward_gpu(THFloatTensor* in_t, THFloatTensor* x_t, THFloatTensor* x_norm_t, THFloatTensor* mean_t, THFloatTensor* rolling_mean_t, THFloatTensor* variance_t, THFloatTensor* rolling_variance_t, THFloatTensor* scales_t, THFloatTensor* biases_t, int train, THFloatTensor* out_t)
void bn_forward_gpu(THCudaTensor* in_t, THCudaTensor* x_t, THCudaTensor* x_norm_t, THCudaTensor* mean_t, THCudaTensor* rolling_mean_t, THCudaTensor* variance_t, THCudaTensor* rolling_variance_t, THCudaTensor* scales_t, THCudaTensor* biases_t, int train, THCudaTensor* out_t)
{
    float * input_gpu = THCudaTensor_data(state, in_t);
    float * output_gpu = THCudaTensor_data(state, out_t);
    float * x_gpu = THCudaTensor_data(state, x_t);
    float * x_norm_gpu = THCudaTensor_data(state, x_norm_t);
    float * mean_gpu = THCudaTensor_data(state, mean_t);
    float * rolling_mean_gpu = THCudaTensor_data(state, rolling_mean_t);
    float * variance_gpu = THCudaTensor_data(state, variance_t);
    float * rolling_variance_gpu = THCudaTensor_data(state, rolling_variance_t);
    float * scales_gpu = THCudaTensor_data(state, scales_t);
    float * biases_gpu = THCudaTensor_data(state, biases_t);
    
    THLongStorage* size_s = THCudaTensor_newSizeOf(state, out_t);
    long * size = THLongStorage_data(size_s);

    long batch = size[0]; 
    long out_c = size[1];
    long out_h = size[2];
    long out_w = size[3];

    copy_ongpu(out_c*out_h*out_w*batch, input_gpu, 1, output_gpu, 1);
    
    if (train) {
        fast_mean_gpu(output_gpu, batch, out_c, out_h*out_w, mean_gpu);
        fast_variance_gpu(output_gpu, mean_gpu, batch, out_c, out_h*out_w, variance_gpu);

        scal_ongpu(out_c, .99, rolling_mean_gpu, 1);
        axpy_ongpu(out_c, .01, mean_gpu, 1, rolling_mean_gpu, 1);
        scal_ongpu(out_c, .99, rolling_variance_gpu, 1);
        axpy_ongpu(out_c, .01, variance_gpu, 1, rolling_variance_gpu, 1);

        copy_ongpu(out_c*out_h*out_w*batch, output_gpu, 1, x_gpu, 1);
        normalize_gpu(output_gpu, mean_gpu, variance_gpu, batch, out_c, out_h*out_w);
        copy_ongpu(out_c*out_h*out_w*batch, output_gpu, 1, x_norm_gpu, 1);
    } else {
        normalize_gpu(output_gpu, rolling_mean_gpu, rolling_variance_gpu, batch, out_c, out_h*out_w);
    }

    scale_bias_gpu(output_gpu, scales_gpu, batch, out_c, out_h*out_w);

    add_bias_gpu(output_gpu, biases_gpu, batch, out_c, out_w*out_h);
}

//void bn_backward_gpu(THFloatTensor* grad_out_t, THFloatTensor* x_t, THFloatTensor* x_norm_t, THFloatTensor* mean_t, THFloatTensor* mean_delta_t, THFloatTensor* variance_t, THFloatTensor* variance_delta_t, THFloatTensor* scales_t,THFloatTensor* scale_delta_t, THFloatTensor* biases_t,THFloatTensor* bias_delta_t,  int train, THFloatTensor* grad_in_t)
void bn_backward_gpu(THCudaTensor* grad_out_t, THCudaTensor* x_t, THCudaTensor* x_norm_t, THCudaTensor* mean_t, THCudaTensor* mean_delta_t, THCudaTensor* variance_t, THCudaTensor* variance_delta_t, THCudaTensor* scales_t,THCudaTensor* scale_delta_t, THCudaTensor* biases_t,THCudaTensor* bias_delta_t,  int train, THCudaTensor* grad_in_t)
{
    float * state_delta_gpu = THCudaTensor_data(state, grad_out_t);
    float * delta_gpu = THCudaTensor_data(state, grad_in_t);
    float * x_gpu = THCudaTensor_data(state, x_t);
    float * x_norm_gpu = THCudaTensor_data(state, x_norm_t);
    float * mean_gpu = THCudaTensor_data(state, mean_t);
    float * mean_delta_gpu = THCudaTensor_data(state, mean_delta_t);
    float * variance_gpu = THCudaTensor_data(state, variance_t);
    float * variance_delta_gpu = THCudaTensor_data(state, variance_delta_t);
    float * scales_gpu = THCudaTensor_data(state, scales_t);
    float * scale_delta_gpu = THCudaTensor_data(state, scale_delta_t);
    float * bias_delta_gpu = THCudaTensor_data(state, bias_delta_t);
   
    THLongStorage* size_s = THCudaTensor_newSizeOf(state, grad_out_t);
    long * size = THLongStorage_data(size_s);

    long batch = size[0]; 
    long out_c = size[1];
    long out_h = size[2];
    long out_w = size[3];

    copy_ongpu(out_c*out_h*out_w*batch, state_delta_gpu, 1, delta_gpu, 1);

    backward_bias_gpu(bias_delta_gpu, delta_gpu, batch, out_c, out_w*out_h);

    backward_scale_gpu(x_norm_gpu, delta_gpu, batch, out_c, out_w*out_h, scale_delta_gpu);

    scale_bias_gpu(delta_gpu, scales_gpu, batch, out_c, out_h*out_w);

    fast_mean_delta_gpu(delta_gpu, variance_gpu, batch, out_c, out_w*out_h, mean_delta_gpu);
    fast_variance_delta_gpu(x_gpu, delta_gpu, mean_gpu, variance_gpu, batch, out_c, out_w*out_h, variance_delta_gpu);
    normalize_delta_gpu(x_gpu, mean_gpu, variance_gpu, mean_delta_gpu, variance_delta_gpu, batch, out_c, out_w*out_h, delta_gpu);
}
#endif
