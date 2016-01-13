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
      this->blobs_.resize(3);
    } else {
      this->blobs_.resize(2);
    }
    // Intialize the weight
    vector<int> weight_shape(1, K_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));
    this->blobs_[2].reset(new Blob<Dtype>(weight_shape));
    caffe_rng_uniform<Dtype>(K_, (Dtype)0, (Dtype)1, this->blobs_[2]->mutable_cpu_data());
    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.circulant_projection_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.circulant_projection_param().bias_filler()));
      bias_filler->Fill(this->blobs_[1].get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  this->param_propagate_down_[2] = false;

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
  this->weight_buffer_.Reshape(weight_shape);
  vector<int> param_shape(1, K_/2+1);
  this->param_buffer_.Reshape(param_shape);
  vector<int> data_shape(2);
  data_shape[0] = M_;
  data_shape[1] = K_;
  this->data_buffer_.Reshape(data_shape);
  vector<int> conv_shape(2);
  conv_shape[0] = M_;
  conv_shape[1] = K_/2+1;
  this->conv_buffer_.Reshape(conv_shape);
  // vector<int> diff_shape(2);
  // diff_shape[0] = M_;
  // diff_shape[1] = N_/2+1;
  // this->diff_buffer_.Reshape(conv_shape);
  LOG(INFO)<<"Buffer Allocated: "<<weight_shape[0]<<"x"<<weight_shape[1];
  // Set up the bias multiplier
  vector<int> bias_shape(1, M_);
  bias_multiplier_.Reshape(bias_shape);
  caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
}

template <typename Dtype>
inline Dtype CirculantProjectionLayer<Dtype>::getFlipInput(const Dtype* input, int index) {
  const Dtype* flip_data = this->blobs_[2]->cpu_data();
  
  if(flip_data[index] > (Dtype)0.5)
    return input[index];
  else
    return -input[index];
}

template <typename Dtype>
void CirculantProjectionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();
  const Dtype* weight = this->blobs_[0]->cpu_data();
  complex<Dtype>* conv_buffer = (this->conv_buffer_.mutable_cpu_data());
  complex<Dtype>* param_buffer = (this->param_buffer_.mutable_cpu_data()); 
  Dtype* data_buffer = this->data_buffer_.mutable_cpu_data();
  int Kc = K_ / 2 + 1;
  
  LOG(INFO)<<"Forward/Flip";
  for(int i=0; i<M_; i++)
    for(int j=0; j<K_; j++)
      (data_buffer + i*K_)[j] = this->getFlipInput(bottom_data + i*K_, j);
        
  LOG(INFO)<<"Forward/FFT";
  caffe_cpu_fft<Dtype>(1, K_, weight, param_buffer);
  caffe_cpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
  LOG(INFO)<<"Forward/MUL";
  for(int i=0; i<M_; i++)
  {
    caffe_mul<complex<Dtype>>(Kc, param_buffer, conv_buffer + i*Kc, conv_buffer + i*Kc);
  }
  LOG(INFO)<<"FORWARD/IFFT";
  caffe_cpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
  for(int i=0; i<M_; i++)
  {
    caffe_copy<Dtype>(N_, data_buffer + i*K_, top_data + i*N_);
  }
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
    complex<Dtype>* conv_buffer = this->conv_buffer_.mutable_cpu_data();
    complex<Dtype>* diff_buffer = this->conv_buffer_.mutable_cpu_diff();
    Dtype* data_buffer = this->data_buffer_.mutable_cpu_data(); 
    int Kc = K_ / 2 + 1;
 
    // Gradient with respect to weight
  
    for(int i=0; i<M_; i++)
    {
      caffe_copy<Dtype>(N_, top_diff + i*N_, data_buffer + i*K_);
      for(int j=N_; j<K_; j++) (data_buffer + i*K_)[j] = (Dtype)0;
    }
    caffe_cpu_fft<Dtype>(M_, K_, data_buffer, diff_buffer);
    for(int i=0; i<M_; i++)
      for(int j=0; j<K_; j++)
        (data_buffer + i*K_)[(K_-j)%K_] = this->getFlipInput(bottom_data + i*K_, j);
    caffe_cpu_fft<Dtype>(M_, K_, data_buffer, conv_buffer);
    caffe_mul<complex<Dtype> >(M_ * Kc, conv_buffer, diff_buffer, conv_buffer);
    caffe_cpu_ifft<Dtype>(M_, K_, conv_buffer, data_buffer);
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, K_, (Dtype)1., data_buffer,
			  bias_multiplier_.cpu_data(), (Dtype)0.,
			  this->blobs_[0]->mutable_cpu_diff());
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
      for(int j=0; j<K_; j++)
	(weight_buffer + i*K_)[j]=this->blobs_[2]->cpu_data()[j]>(Dtype)0.5?param_buffer[(K_+i-j)%K_]:-param_buffer[(K_+i-j)%K_];
    // Gradient with respect to bottom data
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
        top_diff, weight_buffer, (Dtype)0.,
        bottom[0]->mutable_cpu_diff());
  }
}

#ifdef CPU_ONLY
STUB_GPU(CirculantProjectionLayer);
#endif

INSTANTIATE_CLASS(CirculantProjectionLayer);
REGISTER_LAYER_CLASS(CirculantProjection);

}  // namespace caffe
