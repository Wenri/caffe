#include <vector>
#include <boost/make_shared.hpp>

#include "caffe/filler.hpp"
#include "caffe/layers/kronecker_product_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void KroneckerProductLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const int num_output = this->layer_param_.kronecker_product_param().num_output();
  bias_term_ = this->layer_param_.kronecker_product_param().bias_term();
  // N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.kronecker_product_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N kronecker products with dimension CHW are performed.
  // K_ = bottom[0]->count(axis);
  this->dims_.push_back(std::make_pair(5,32));
  this->dims_.push_back(std::make_pair(2,32));
 
  N_ = K_ = 1;
  for(const auto& dim : this->dims_) {
    N_ = N_ * dim.first;
    K_ = K_ * dim.second;
  }

  CHECK_EQ(K_, bottom[0]->count(axis)) << "K mismatch!";
  CHECK_EQ(N_, num_output) << "N mismatch!";
  
  // Check if we need to set up the weights
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    // Intialize the weight
    for(const auto& dim : this->dims_) {
      vector<int> weight_shape(2);
      weight_shape[0] = dim.first;
      weight_shape[1] = dim.second;
      auto weight = boost::make_shared<Blob<Dtype> >(weight_shape);
      this->blobs_.push_back(weight);
      // fill the weights
      shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
          this->layer_param_.kronecker_product_param().weight_filler()));
      weight_filler->Fill(weight.get());
    }
    // If necessary, intiialize and fill the bias term
    if (bias_term_) {
      vector<int> bias_shape(1, N_);
      auto weight = boost::make_shared<Blob<Dtype> >(bias_shape);
      shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
          this->layer_param_.kronecker_product_param().bias_filler()));
      this->blobs_.push_back(weight);
      bias_filler->Fill(weight.get());
    }
  }  // parameter initialization
  this->param_propagate_down_.resize(this->blobs_.size(), true);
  this->unitTest();
  LOG(INFO)<<"Init Succeed " << N_ << "x" << K_;
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {

  LOG(INFO)<<"Reshpae";
  
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.kronecker_product_param().axis());
  const int new_K = bottom[0]->count(axis);
  CHECK_EQ(K_, new_K)
      << "Input size incompatible with kronecker product parameters.";
  // The first "axis" dimensions are independent kronecker products; the total
  // number of these is M_, the product over these dimensions.
  M_ = bottom[0]->count(0, axis);
  // The top shape will be the bottom shape with the flattened axes dropped,
  // and replaced by a single axis with dimension num_output (N_).
  vector<int> top_shape = bottom[0]->shape();
  top_shape.resize(axis + 1);
  top_shape[axis] = N_;
  top[0]->Reshape(top_shape);

  vector<int> buf_shape(1, M_ * K_);
  this->kpbuf_.Reshape(buf_shape);
  // Set up the bias multiplier
  if (bias_term_) {
    vector<int> bias_shape(1, M_);
    bias_multiplier_.Reshape(bias_shape);
    caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  }

  LOG(INFO)<<"Reshape Done. " << M_;
}

template <typename Dtype>
bool KroneckerProductLayer<Dtype>::unitTest()
{
  Dtype utA[] = {1,-1,1,
		 -2,2,-2};
  Dtype utB[] = {1,2,3,
		 4,5,6};
  Dtype utx[] = {9,8,7,6,5,4,3,2,1};
  Dtype top[] = {1,1,1,1};
  Dtype result1[100] = {0,};
  Dtype result2[100] = {0,};
  Dtype result3[100] = {0,};
  Dtype result4[100] = {0,};
	
  vector<int> buf_shape(1, 100);
  
  this->kpbuf_.Reshape(buf_shape);
  this->akpbx(utA, utB, utx, result1, 2, 3, 2, 3);
  LOG(INFO)<<"UNITTEST-FWD: "<<result1[0]<<" "
	   <<result1[1]<<" "
	   <<result1[2]<<" "
	   <<result1[3];

  this->dakpbx(top, utB, utx, result2, 2, 3, 2, 3);
  LOG(INFO)<<"UNITTEST-DA: "<<result2[0]<<" "
	   <<result2[1]<<" "
	   <<result2[2]<<" "
	   <<result2[3]<<" "
	   <<result2[4]<<" "
	   <<result2[5];

  this->akpdbx(top, utA, utx, result3, 2, 3, 2, 3);
  LOG(INFO)<<"UNITTEST-DB: "<<result3[0]<<" "
	   <<result3[1]<<" "
	   <<result3[2]<<" "
	   <<result3[3]<<" "
	   <<result3[4]<<" "
	   <<result3[5];

 this->akpbdx(top, utA, utB, result4, 2, 3, 2, 3);
  LOG(INFO)<<"UNITTEST-Dx: "<<result4[0]<<" "
	   <<result4[1]<<" "
	   <<result4[2]<<" "
	   <<result4[3]<<" "
	   <<result4[4]<<" "
	   <<result4[5]<<" "
	   <<result4[6]<<" "
	   <<result4[7]<<" "
	   <<result4[8];

  
  return true;
}
  
template <typename Dtype>
void KroneckerProductLayer<Dtype>::akpbx(const Dtype* A, const Dtype* B, const Dtype* x, Dtype *t, const int k1, const int d1, const int k2, const int d2) {

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, k1, d2, d1, (Dtype)1.,
			A, x, (Dtype)0., this->kpbuf_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k1, k2, d2, (Dtype)1.,
			this->kpbuf_.cpu_data(), B, (Dtype)0., t);
  
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::dakpbx(const Dtype* top, const Dtype* B, const Dtype* x, Dtype *t, const int k1, const int d1, const int k2, const int d2) {

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, k2, d1, d2, (Dtype)1.,
			B, x, (Dtype)0., this->kpbuf_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, k1, d1, k2, (Dtype)1.,
			top, this->kpbuf_.cpu_data(), (Dtype)1., t);
  
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::akpdbx(const Dtype* top, const Dtype* A, const Dtype* x, Dtype *t, const int k1, const int d1, const int k2, const int d2) {

  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, k1, d2, d1, (Dtype)1.,
			A, x, (Dtype)0., this->kpbuf_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, k2, d2, k1, (Dtype)1.,
			top, this->kpbuf_.cpu_data(), (Dtype)1., t);
  
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::akpbdx(const Dtype* top, const Dtype* A, const Dtype* B, Dtype *t, const int k1, const int d1, const int k2, const int d2) {

  caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, d1, k2, k1, (Dtype)1.,
			A, top, (Dtype)0., this->kpbuf_.mutable_cpu_data());
  caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, d1, d2, k2, (Dtype)1.,
			this->kpbuf_.cpu_data(), B, (Dtype)0., t);
  
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  const Dtype* bottom_data = bottom[0]->cpu_data();
  Dtype* top_data = top[0]->mutable_cpu_data();

  // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasTrans, M_, N_, K_, (Dtype)1.,
  //     bottom_data, weight, (Dtype)0., top_data);
  LOG(INFO)<<"Forward";
  
  for (int i=0; i<M_; i++)
    this->akpbx(this->blobs_[0]->cpu_data(), this->blobs_[1]->cpu_data(),
	  (bottom_data + i*K_), (top_data + i*N_),
	  this->dims_[0].first, this->dims_[0].second,
	  this->dims_[1].first, this->dims_[1].second);

  if (bias_term_) {
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void KroneckerProductLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {

  LOG(INFO)<<"Backward";
  
  if (this->param_propagate_down_[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    
    // Gradient with respect to weight
    // caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
    //     top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    caffe_set(this->dims_[0].first * this->dims_[0].second,
	      (Dtype)0, this->blobs_[0]->mutable_cpu_diff());
    for (int i=0; i<M_; i++)
      this->dakpbx((top_diff + i*N_), this->blobs_[1]->cpu_data(),
	  (bottom_data + i*K_), this->blobs_[0]->mutable_cpu_diff(),
	  this->dims_[0].first, this->dims_[0].second,
	  this->dims_[1].first, this->dims_[1].second);

  }
  if (this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    const Dtype* bottom_data = bottom[0]->cpu_data();
    // Gradient with respect to weight
    // caffe_cpu_gemm<Dtype>(CblasTrans, CblasNoTrans, N_, K_, M_, (Dtype)1.,
    //     top_diff, bottom_data, (Dtype)1., this->blobs_[0]->mutable_cpu_diff());
    caffe_set(this->dims_[1].first * this->dims_[1].second,
	      (Dtype)0, this->blobs_[1]->mutable_cpu_diff());
    for (int i=0; i<M_; i++)
      this->akpdbx((top_diff + i*N_), this->blobs_[0]->cpu_data(),
	  (bottom_data + i*K_), this->blobs_[1]->mutable_cpu_diff(),
	  this->dims_[0].first, this->dims_[0].second,
	  this->dims_[1].first, this->dims_[1].second);

  }
  if (bias_term_ && this->param_propagate_down_[this->dims_.size()]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
	this->blobs_[this->dims_.size()]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
    // Gradient with respect to bottom data
    // caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, K_, N_, (Dtype)1.,
    //     top_diff, this->blobs_[0]->cpu_data(), (Dtype)0.,
    //     bottom[0]->mutable_cpu_diff());
    for (int i=0; i<M_; i++) {
      this->akpbdx(
		    (top_diff + i*N_),
		    this->blobs_[0]->cpu_data(),
		    this->blobs_[1]->cpu_data(),
		    (bottom_diff + i*K_),
		    this->dims_[0].first, this->dims_[0].second,
		    this->dims_[1].first, this->dims_[1].second
		    );
    }
 
  }
}

#ifdef CPU_ONLY
STUB_GPU(KroneckerProductLayer);
#endif

INSTANTIATE_CLASS(KroneckerProductLayer);
REGISTER_LAYER_CLASS(KroneckerProduct);

}  // namespace caffe
