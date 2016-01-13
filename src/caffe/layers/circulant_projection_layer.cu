#include <vector>
#include <thrust/complex.h>
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
__global__ void bat_sgnflp_knl(const int m, const int n, const Dtype* r, Dtype* t, const Dtype* d) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    int off = batch * n;
    (t + off)[index] = getFlipInput(r + off, index, d);
  }
}

template <typename Dtype>
__global__ void bat_mul_knl(const int m, const int n, const Dtype* r, Dtype* t, const Dtype* d) {
  CUDA_KERNEL_LOOP_2D(batch, index, m, n) {
    int off = batch * n;
    (t + off)[index] = (r + off)[index] * d[index];
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {

  const Dtype* bottom_data = bottom[0]->gpu_data();
  Dtype* top_data = top[0]->mutable_gpu_data();
  const Dtype* weight = this->blobs_[0]->gpu_data();
  complex<Dtype>* conv_buffer = this->conv_buffer_.mutable_gpu_data();
  complex<Dtype>* param_buffer = this->param_buffer_.mutable_gpu_data(); 
  Dtype* weight_buffer = this->weight_buffer_.mutable_gpu_data();
  Dtype* data_buffer = this->data_buffer_.mutable_gpu_data();
  int Kc = K_ / 2 + 1;
  
  bat_sgnflp_knl<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, K_, bottom_data, data_buffer, this->blobs_[2]->gpu_data());
  CUDA_POST_KERNEL_CHECK;
  
  LOG(INFO)<<"Forward/GPU_FFT";
  caffe_gpu_fft<Dtype>(1, K_, weight, param_buffer);
  caffe_gpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
  LOG(INFO)<<"Forward/MUL";

  bat_mul_knl<thrust::complex<Dtype> >
    <<<CAFFE_GET_BLOCKS_2D(M_, Kc), CAFFE_CUDA_NUM_THREADS_2D>>>
    (M_, Kc,
     reinterpret_cast<thrust::complex<Dtype> *>(conv_buffer),
     reinterpret_cast<thrust::complex<Dtype> *>(conv_buffer),
     reinterpret_cast<thrust::complex<Dtype> *>(param_buffer)
     );
  CUDA_POST_KERNEL_CHECK;
  
  LOG(INFO)<<"FORWARD/IFFT";
  caffe_gpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
  CUDA_CHECK(cudaMemcpy2D(top_data, N_ * sizeof(Dtype),
			  data_buffer, K_ * sizeof(Dtype),
			  N_ * sizeof(Dtype), M_, cudaMemcpyDeviceToDevice));
 
  if (bias_term_) {
    caffe_gpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.gpu_data(),
        this->blobs_[1]->gpu_data(), (Dtype)1., top_data);
  }

}

template <typename Dtype>
__global__ void bat_cirflp_knl(const int k, const int n, Dtype* a, const Dtype* b, const Dtype* flip) {
  CUDA_KERNEL_LOOP_2D(batch, index, k, n) {
    (a + batch*n)[(n-index)%n]=getFlipInput(b + batch*n, index, flip);
  }
}

template <typename Dtype>
__global__ void circpy_knl(const int n, Dtype* dist, const Dtype* src) {
  CUDA_KERNEL_LOOP(i, n) {
    dist[(n-i)%n] = src[i];
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  const Dtype* top_diff = top[0]->gpu_diff();
  const int Kc = K_ / 2 + 1;
  complex<Dtype>* diff_buffer;

  if (this->param_propagate_down_[0] || propagate_down[0] ){
    Dtype* data_buffer = this->data_buffer_.mutable_gpu_data();
    
    diff_buffer = this->conv_buffer_.mutable_gpu_diff();
    
    CUDA_CHECK(cudaMemcpy2D(data_buffer, K_ * sizeof(Dtype),
			    top_diff, N_ * sizeof(Dtype),
			    N_ * sizeof(Dtype), M_, cudaMemcpyDeviceToDevice));
    if (N_ < K_)
      CUDA_CHECK(cudaMemset2D(data_buffer + N_, K_* sizeof(Dtype),
			      0, (K_ - N_) * sizeof(Dtype), M_));
    
    caffe_gpu_fft<Dtype>(M_, K_, data_buffer, diff_buffer);    
  }
  if (this->param_propagate_down_[0]) {
    const Dtype* bottom_data = bottom[0]->gpu_data();
    complex<Dtype>* conv_buffer = this->conv_buffer_.mutable_gpu_data();
    Dtype* data_buffer = this->data_buffer_.mutable_gpu_data();
 
    LOG(INFO)<<"Backward/FFT";

    bat_cirflp_knl<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
      (M_, K_, data_buffer, bottom_data, this->blobs_[2]->gpu_data());
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
    caffe_gpu_mul<complex<Dtype> >(M_ * Kc, conv_buffer, diff_buffer, conv_buffer);
    caffe_gpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, K_, (Dtype)1., data_buffer,
			  bias_multiplier_.gpu_data(), (Dtype)0.,
			  this->blobs_[0]->mutable_gpu_diff());
  }
  if (propagate_down[0]) {
    complex<Dtype>* param_buffer = this->param_buffer_.mutable_gpu_data();
    Dtype* weight_buffer = this->weight_buffer_.mutable_gpu_data();
    Dtype* data_buffer = this->data_buffer_.mutable_gpu_data();
    
    circpy_knl<Dtype><<<CAFFE_GET_BLOCKS(K_), CAFFE_CUDA_NUM_THREADS>>>
      (K_,  weight_buffer, this->blobs_[0]->gpu_data());
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_fft<Dtype>(1, K_, weight_buffer, param_buffer);
    bat_mul_knl<thrust::complex<Dtype> >
      <<<CAFFE_GET_BLOCKS_2D(M_, Kc), CAFFE_CUDA_NUM_THREADS_2D>>>
      (M_, Kc,
       reinterpret_cast<thrust::complex<Dtype> *>(diff_buffer),
       reinterpret_cast<thrust::complex<Dtype> *>(diff_buffer),
       reinterpret_cast<thrust::complex<Dtype> *>(param_buffer)
       );
    CUDA_POST_KERNEL_CHECK;
    caffe_gpu_ifft<Dtype>(M_, K_, diff_buffer, data_buffer);
    bat_sgnflp_knl<Dtype><<<CAFFE_GET_BLOCKS_2D(M_, K_), CAFFE_CUDA_NUM_THREADS_2D>>>
      (M_, K_, data_buffer, bottom[0]->mutable_gpu_diff(),
       this->blobs_[2]->gpu_data());
    CUDA_POST_KERNEL_CHECK;
  }

  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->gpu_diff();
    // Gradient with respect to bias
    caffe_gpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.gpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_gpu_diff());
  }
  
}

INSTANTIATE_LAYER_GPU_FUNCS(CirculantProjectionLayer);

}  // namespace caffe
