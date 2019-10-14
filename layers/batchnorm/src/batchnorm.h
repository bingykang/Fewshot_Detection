
void bn_forward(THFloatTensor* in_t, THFloatTensor* x_t, THFloatTensor* x_norm_t, THFloatTensor* mean_t, THFloatTensor* rolling_mean_t, THFloatTensor* variance_t, THFloatTensor* rolling_variance_t, THFloatTensor* scales_t, THFloatTensor* biases_t, int train, THFloatTensor* out_t);

void bn_backward(THFloatTensor* grad_out_t, THFloatTensor* x_t, THFloatTensor* x_norm_t, THFloatTensor* mean_t, THFloatTensor* mean_delta_t, THFloatTensor* variance_t, THFloatTensor* variance_delta_t, THFloatTensor* scales_t,THFloatTensor* scale_delta_t, THFloatTensor* biases_t,THFloatTensor* bias_delta_t,  int train, THFloatTensor* grad_in_t);

void bn_forward_gpu(THCudaTensor* in_t, THCudaTensor* x_t, THCudaTensor* x_norm_t, THCudaTensor* mean_t, THCudaTensor* rolling_mean_t, THCudaTensor* variance_t, THCudaTensor* rolling_variance_t, THCudaTensor* scales_t, THCudaTensor* biases_t, int train, THCudaTensor* out_t);
void bn_backward_gpu(THCudaTensor* grad_out_t, THCudaTensor* x_t, THCudaTensor* x_norm_t, THCudaTensor* mean_t, THCudaTensor* mean_delta_t, THCudaTensor* variance_t, THCudaTensor* variance_delta_t, THCudaTensor* scales_t,THCudaTensor* scale_delta_t, THCudaTensor* biases_t,THCudaTensor* bias_delta_t,  int train, THCudaTensor* grad_in_t);
