#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/circulant_projection_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.circulant_projection_param().num_output();
  bias_term_ = this->layer_param_.circulant_projection_param().bias_term();
  LOG(INFO)<<"CirculantProjectionLayerSetUp, N="<<num_output;
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.circulant_projection_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  // Check if we need to set up the weights
  CHECK_LE(N_, K_) << "Currently only N<=K supported.";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    if (bias_term_) {
      this->blobs_.resize(2);
    } else {
      this->blobs_.resize(1);
    }
    // Intialize the weight
    vector<int> bias_shape(1, N_);
    this->data_flip_.Reshape(bias_shape);
    caffe_rng_uniform<Dtype>(N_, (Dtype)0, (Dtype)1, this->data_flip_.mutable_cpu_data());
    this->blobs_[0].reset(new Blob<Dtype>(bias_shape));
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.circulant_projection_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.circulant_projection_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.circulant_projection_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with inner product parameters.";
  // The first "axis" dimensions are independent inner products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);
  vector<int> weight_shape(2);
  weight_shape[0] = K_;
  weight_shape[1] = K_;
  vector<int> bias_shape(1, K_);
  this->weight_buffer_.Reshape(weight_shape);
  this->param_buffer_.Reshape(bias_shape);
  vector<int> data_shape(2);
  data_shape[0] = M_;
  data_shape[1] = K_;
  this->data_buffer_.Reshape(data_shape);
  this->conv_buffer_.Reshape(data_shape);
  LOG(INFO)<<"Buffer Allocated: "<<weight_shape[0]<<"x"<<weight_shape[1];
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  std::complex<Dtype>* conv_buffer = reinterpret_cast<std::complex<Dtype>*>(this->conv_buffer_.mutable_cpu_data());
  std::complex<Dtype>* param_buffer = reinterpret_cast<std::complex<Dtype>*>(this->param_buffer_.mutable_cpu_data()); 

  LOG(INFO)<<"Forward/FFT";
  // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
  //     bottom_data, weight, (Dtype)0., top_data);
  caffe_cpu_fft<Dtype>(1, N_, weight, param_buffer);
  caffe_cpu_fft<Dtype>(M_, N_, bottom_data, conv_buffer);
  LOG(INFO)<<"Forward/MUL";
  for(int i=0; i<M_; i++)
  {
    caffe_mul<std::complex<Dtype>>(N_, param_buffer, conv_buffer + i*N_, conv_buffer + i*N_);
  }
  LOG(INFO)<<"FORWARD/IFFT";
  caffe_cpu_ifft<Dtype>(M_, N_, conv_buffer, top_data);

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    std::complex<Dtype>* conv_buffer = reinterpret_cast<std::complex<Dtype>*>(this->conv_buffer_.mutable_cpu_data());
    std::complex<Dtype>* param_buffer = reinterpret_cast<std::complex<Dtype>*>(this->param_buffer_.mutable_cpu_data()); 
    Dtype* weight_buffer = this->weight_buffer_.mutable_cpu_data();
    Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
    // Gradient with respect to weight
    // caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
    //     top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    LOG(INFO)<<"Backword";
    caffe_cpu_fft<Dtype>(M_, N_, top_diff, conv_buffer);
    for(int i=0; i<M_; i++)
    {
      for(int j=0; j<K_; j++) weight_buffer[(K_-j)%K_]=bottom_data[i*K_+j];
      caffe_cpu_fft<Dtype>(1, K_, weight_buffer, param_buffer);
      caffe_mul<std::complex<Dtype>>(N_, conv_buffer + i*N_, param_buffer, param_buffer);
      caffe_cpu_ifft<Dtype>(1, K_, param_buffer, weight_buffer);
      caffe_add<Dtype>(N_, weight_diff, weight_buffer, weight_diff);
    }
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* param_buffer = this->blobs_[0]->cpu_data();
    Dtype* weight_buffer = this->weight_buffer_.mutable_cpu_data();
    for(int i=0; i<K_; i++)
      for(int j=0; j<N_; j++)
	weight_buffer[i*N_+j]=param_buffer[(N_+i-j)%N_];
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, this->weight_buffer_.cpu_data(), (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(CirculantProjectionLayer);
#endif

INSTANTIATE_CLASS(CirculantProjectionLayer);
REGISTER_LAYER_CLASS(CirculantProjection);

}  // namespace caffe
