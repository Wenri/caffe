#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/circulant_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__device__ inline Dtype getFlipInput(const Dtype* input, int index, const Dtype* flip_data) {
  if(flip_data[index] > (Dtype)0.5)
    return input[index];
  else
    return -input[index];
}

template <typename Dtype>
__global__ void sign_flip_kernel(const int m, const int n, const Dtype* r, Dtype* t, const Dtype* d) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    int off = batch * n;
    (t + off)[index] = getFlipInput(r + off, index, d);
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  std::complex<Dtype>* conv_buffer = this->conv_buffer_.mutable_gpu_data();
  std::complex<Dtype>* param_buffer = this->param_buffer_.mutable_gpu_data(); 
  Dtype* weight_buffer = this->weight_buffer_.mutable_gpu_data();
  Dtype* data_buffer = this->data_buffer_.mutable_gpu_data();
  int Kc = K_ / 2 + 1;
  
  sign_flip_kernel<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, K_, bottom_data, data_buffer, this->blobs_[2]->gpu_data());

  LOG(INFO)<<"Forward/GPU_FFT";
  caffe_gpu_fft<Dtype>(1, K_, weight, param_buffer);
  caffe_gpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
  LOG(INFO)<<"Forward/MUL";

  for(int i=0; i<M_; i++)
  {
    caffe_gpu_mul<std::complex<Dtype>>(Kc, param_buffer, conv_buffer + i*Kc, conv_buffer + i*Kc);
  }

  LOG(INFO)<<"FORWARD/IFFT";
  caffe_gpu_ifft<Dtype>(M_, K_, conv_buffer, top_data);

  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }

}

template <typename Dtype>
__global__ void circulant_copy_kernel(const int n, Dtype* a, const Dtype* b) {
  CUDA_KERNEL_LOOP(index, n) {
    a[(n-index)%n]=b[index];
  }
}

template <typename Dtype>
__global__ void circulant_matrix_kernel(const int k, const int n, Dtype* dist, const Dtype* src, const Dtype* flip) {
  CUDA_KERNEL_LOOP_2D(i, j, k, n) {
    (dist + i*n)[j] = getFlipInput(src, (n+i-j)%n, flip);
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* bottom_data = bottom[0]->gpu_data();
    std::complex<Dtype>* conv_buffer = this->conv_buffer_.mutable_gpu_data();
    std::complex<Dtype>* param_buffer = this->param_buffer_.mutable_gpu_data(); 
    Dtype* weight_buffer = this->weight_buffer_.mutable_gpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_gpu_diff();
    Dtype* data_buffer = this->data_buffer_.mutable_gpu_data();
    int Nc = N_ / 2 + 1;
    int Kc = K_ / 2 + 1;
    LOG(INFO)<<"Backword/FFT-"<<top_diff<<"-"<<conv_buffer;
    caffe_gpu_fft<Dtype>(M_, N_, top_diff, conv_buffer);

    sign_flip_kernel<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
      (M_, K_, bottom_data, data_buffer, this->blobs_[2]->gpu_data());

    LOG(INFO)<<"Backword/MUL-IFFT";
    for(int i=0; i<M_; i++)
    {
      circulant_copy_kernel<Dtype><<<CAFFE_GET_BLOCKS(K_), CAFFE_CUDA_NUM_THREADS>>>(
										     K_,
									      weight_buffer,
									      data_buffer+i*K_);
      caffe_gpu_fft<Dtype>(1, K_, weight_buffer, param_buffer);
      caffe_gpu_mul<std::complex<Dtype>>(Kc, conv_buffer + i*Kc, param_buffer, param_buffer);
      caffe_gpu_ifft<Dtype>(1, K_, param_buffer, weight_buffer);
      caffe_gpu_add<Dtype>(N_, weight_diff, weight_buffer, weight_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    const Dtype* param_buffer = this->blobs_[0]->gpu_data();
    Dtype* weight_buffer = this->weight_buffer_.mutable_gpu_data();

    circulant_matrix_kernel<Dtype><<<CAFFE_GET_BLOCKS_2D(K_, N_), CAFFE_CUDA_NUM_THREADS_2D>>>(
											       K_,
											       N_,
						 				       weight_buffer,
										       param_buffer,
								        this->blobs_[2]->gpu_data());
    // Gradient with respect to bottom data
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->weight_buffer_.gpu_data(), (Dtype)0.,
        bottom[0]->mutable_gpu_diff());
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(CirculantProjectionLayer);

}  // namespace caffe
