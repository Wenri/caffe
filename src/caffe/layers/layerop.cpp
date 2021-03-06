#include <vector>

#include "caffe/filler.hpp"
#include "caffe/layers/layerop.hpp"
#include "caffe/util/math_functions.hpp"
#include "structured/lib/ProcessorTape.h"
#include "structured/interface/caffe/TypedData_Caffe.h"
#include "structured/interface/caffe/layerop.hpp"
#include "structured/lib/AdjointPoint.h"

using namespace structured;

namespace caffe {

template <typename Dtype>
void LayerOpLayer<Dtype>::initParams() {
    // Intialize the weight
    vector<int> weight_shape(1, K_);
    this->blobs_[0].reset(new Blob<Dtype>(weight_shape));

    // fill the weights
    shared_ptr<Filler<Dtype> > weight_filler(GetFiller<Dtype>(
        this->layer_param_.layer_op_param().weight_filler()));
    weight_filler->Fill(this->blobs_[0].get());

    this->param_propagate_down_[0] = true;

    vector<int> bias_shape(1, N_);
    this->blobs_[1].reset(new Blob<Dtype>(bias_shape));
    shared_ptr<Filler<Dtype> > bias_filler(GetFiller<Dtype>(
	  this->layer_param_.layer_op_param().bias_filler()));
    bias_filler->Fill(this->blobs_[1].get());

    this->param_propagate_down_[1] = true;
}
  
template <typename Dtype>
void LayerOpLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  ProcessorRepresentative<Dtype> representative;
  Environment env;
  const int num_output = this->layer_param_.layer_op_param().num_output();
  
  this->processor.reset(representative(&env));
  
  bias_term_ = this->layer_param_.layer_op_param().bias_term();
  flip_term_ = true;
  N_ = num_output;
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.layer_op_param().axis());
  // Dimensions starting from "axis" are "flattened" into a single
  // length K_ vector. For example, if bottom[0]'s shape is (N, C, H, W),
  // and axis == 1, N inner products with dimension CHW are performed.
  K_ = bottom[0]->count(axis);
  LOG(INFO)<<"LayerOpLayerSetUp, N="<<num_output<<", K="<<K_;
  // Check if we need to set up the weights
  CHECK_LE(N_, K_) << "Currently only N<=K supported.";
  if (this->blobs_.size() > 0) {
    LOG(INFO) << "Skipping parameter initialization";
  } else {
    this->blobs_.resize(2);
    this->param_propagate_down_.resize(this->blobs_.size(), false);
    this->initParams();
  }  // parameter initialization
}

template <typename Dtype>
void LayerOpLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  // Figure out the dimensions
  const int axis = bottom[0]->CanonicalAxisIndex(
      this->layer_param_.layer_op_param().axis());
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
  // Set up the bias multiplier
  vector<int> bias_shape(1, M_);
  bias_multiplier_.Reshape(bias_shape);
  caffe_set(M_, Dtype(1), bias_multiplier_.mutable_cpu_data());
  
}

template <typename Dtype>
void LayerOpLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  ProcessorTape atape;
  CpuCoreCaffe core(this->processor.get());
  LOG(INFO)<<"Fwd Method called.\n";
  for(auto in : bottom){
    atape.input.push_back( std::shared_ptr< BufferedData > (
	       new TypedDataCaffeCpu<Dtype>(*in)
    ));
  }
  for(auto in : this->blobs_){
    atape.input.push_back( std::shared_ptr< BufferedData > (
	       new TypedDataCaffeCpu<Dtype>(*in)
    ));
  }
  for(auto out : top){
    atape.output.push_back( std::shared_ptr< BufferedData > (
	       new TypedDataCaffeCpu<Dtype>(*out)
    ));
  }
  functor::LayerOpFunctor<CPUDevice>()(CPUDevice(), &core, &atape);
  if (bias_term_) {
    Dtype* top_data = top[0]->mutable_cpu_data();
    caffe_cpu_gemm<Dtype>(CblasNoTrans, CblasNoTrans, M_, N_, 1, (Dtype)1.,
        bias_multiplier_.cpu_data(),
        this->blobs_[1]->cpu_data(), (Dtype)1., top_data);
  }
}

template <typename Dtype>
void LayerOpLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down,
    const vector<Blob<Dtype>*>& bottom) {
  ProcessorTape atape;
  ProcessorTape btape;
  CpuCoreCaffe core(this->processor.get());
  LOG(INFO)<<"Back Method called.\n";
  TypedDataCaffeCpu<Dtype> * typedData;
  for(auto in : bottom){
    typedData = new TypedDataCaffeCpu<Dtype>(*in);
    atape.input.push_back(std::shared_ptr<BufferedData>(typedData));
    typedData = new TypedDataCaffeCpu<Dtype>(*in);
    typedData->swapBuffers();
    btape.output.push_back(std::shared_ptr<BufferedData>(typedData));
  }
  for(auto in : this->blobs_){
    typedData = new TypedDataCaffeCpu<Dtype>(*in);
    atape.input.push_back(std::shared_ptr<BufferedData>(typedData));
    typedData = new TypedDataCaffeCpu<Dtype>(*in);
    typedData->swapBuffers();
    btape.output.push_back(std::shared_ptr<BufferedData>(typedData));
  }
  for(auto out : top){
    typedData = new TypedDataCaffeCpu<Dtype>(*out);
    atape.output.push_back(std::shared_ptr<BufferedData>(typedData));
    typedData = new TypedDataCaffeCpu<Dtype>(*out);
    typedData->swapBuffers();
    btape.input.push_back(std::shared_ptr<BufferedData>(typedData));
  }
  functor::LayerOpFunctor<CPUDevice>()(CPUDevice(), &core, &atape, &btape);
  
  if (this->param_propagate_down_[0]) {
    // Gradient with respect to weight
  }
  if (bias_term_ && this->param_propagate_down_[1]) {
    const Dtype* top_diff = top[0]->cpu_diff();
    // Gradient with respect to bias
    caffe_cpu_gemv<Dtype>(CblasTrans, M_, N_, (Dtype)1., top_diff,
        bias_multiplier_.cpu_data(), (Dtype)1.,
        this->blobs_[1]->mutable_cpu_diff());
  }
  if (propagate_down[0]) {
    // Gradient with respect to bottom data
  }
}

#ifdef CPU_ONLY
STUB_GPU(LayerOpLayer);
#endif

INSTANTIATE_CLASS(LayerOpLayer);
REGISTER_LAYER_CLASS(LayerOp);

}  // namespace caffe
